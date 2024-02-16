"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 由于这个代码使用的是将embedding切分成n_head份实现multi head，因此需要n_head的大小整除embedding的大小
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # 使用tril得到mask矩阵
            # mask的部分，tril是得到一个矩阵的下三角矩阵，把一个全1的矩阵放进去就可以得到mask矩阵。
            # 而register_buffer可以将一个张量注册为一个buffer。
            # 这意味着这个张量不会被计算梯度，也不会对模型的参数进行更新，可以节省内存，提高计算效率。
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 把输入经过线性层得到的矩阵切分，得到qkv三个矩阵
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # 再将k，q，v的embedding分给每个head
        # 矩阵的形状变化：(batch_size, sequence_length, embedding_dimensionality) ->
        # (batch_size, n_head, sequence_length, embedding_dimensionality // n_head)
        # k和q负责计算attention score，v是每个token的embedding
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # 计算了每一对token的Q和K的缩放点积，从而得到每对token之间的attention score
            # att矩阵形状： (batch_size, n_head, sequence_length, sequence_length)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # mask操作，使得每个token和它自己之后token计算出来的attention score被mask掉，从而不会让前面的token得到后面token的相关信息
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 归一化得到0-1之间的score
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 通过attention score乘上每个token的embedding，
            # 每个token的embedding为它所在的sequence embedding的加权和，这样每个embedding都得到了关于整个句子的信息
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 最后把n个head连起来，恢复成(batch_size, sequence_length, embedding_dimensionality)的形式
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# 这个Block的输入的shape和输出的shape都是(batch_size, suquence_length, embedding_dimensionality)
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 一个transformer decoder block的实现, LayerNorm->Attention->LayerNorm->Feed Forward的顺序。
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    # __init__部分中实现了input和output的embedding的参数共享。
    # 具体来说，wte的weight是一个大小为(vocab_size, embedding_size)大小的矩阵，而Linear层的weight就是A矩阵，
    # 为了实现dimension从in_feature到out_feature的转换，A矩阵的形状需要是(out_feature, in_feature)，
    # 刚好就是shape为(vocab_size, embedding_size)的一个矩阵，所以这里直接赋值，就是浅拷贝，实现了参数共享。
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # 一个ModuleDict，包含了很多子层
        # wte：word to embedding，把token转换成embedding
        # wpe：word position embedding，把位置信息转换成embedding
        # drop：dropout层，防止过拟合
        # h：一个ModuleList，包含了n_layer个Block，实现transformer中的多层的结构
        # ln_f：一个layernorm层，进行归一化
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # 这里input和output的embedding使用了参数共享
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 得到一个表示位置的tensor，里面的value是(0, 1, 2, ..., t - 1)
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # 这个地方虽然两个tensor形状不相等，但是因为b是1的倍数，会自动把(1, t, n_embd)进行repeat，得到(b, t, n_embd)，加起来
        x = self.transformer.drop(tok_emb + pos_emb)
        # 经过n_layer层block
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 此时，每一个token的embedding都是包括它在内的它之前的token的某种复杂的加权值，
            # 它学习到了这个前缀序列的信息，我们用预测它的下一个token是什么的这个任务来训练，
            # 用x中的embedding进行decode，把它映射到vocab中去，就得到了每个前缀序列对下一个token的预测值。
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 在inference的时候不需要计算loss，只保留最后一个学习到整个句子的信息
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    # 减少block size的，因为from pretrained的GPT默认block size为1024，这个函数可以减少block size，
    # 目前的block size只在self attention的mask矩阵bias，wpe中用到，所以只用改这几个位置。
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    # from_pretrained的代码，从huggingface的GPT模型中加载weight。
    # 这里比较有意思的是对一个tensor进行copy_的操作也会被记录在计算图上，因此需要使用with torch.no_grad()
    # classmethod表示这是一个类方法，可以通过类名调用，而不用实例化类
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 在这个dict里根据model_type选择需要的type对应的参数
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # 判断是否有dropout的override参数，如果有就更新config_args里的dropout值
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        # 初始化GPTConfig类
        config = GPTConfig(**config_args)
        model = GPT(config)
        # 得到我们自己写的这个GPT模型的所有参数
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # 初始化huggingface的模型
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # 拿出需要copy的参数的key
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # openai的实现里有一些Linear用了Conv1D，导致了weight和我们使用的相比是转置的，把这一部分特别提取出来
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # 如果k是w中某个字符串后缀，那么就copy它的转置
            # endswith是判断某个字符串是不是后缀，tensor.t()是矩阵的转置
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            # 否则就直接copy
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # 配置并返回optimizer，指定在一些权重上衰减或者不衰减。
    # 权重衰减（weight
    # decay）是一种正则化技巧，主要用于防止过拟合。在训练过程中，权重衰减会使得模型的权重变得更小，从而减少模型的复杂度。
    # 在这个代码中，作者将模型中的参数分为两组，一组是需要权重衰减的参数，另一组是不需要权重衰减的参数。参数的分组依据是：
    # 偏置项（bias）参数不需要权重衰减，因为偏置项不参与计算，而且往往很小，所以不需要权重衰减来降低其复杂度。
    # 层归一化（LayerNorm）权重参数也不需要权重衰减，因为它的作用是对输入数据进行标准化，不会对模型的复杂度产生影响。
    # 嵌入（Embedding）参数也不需要权重衰减，因为权重衰减可能会抹除词向量之间的关系，从而降低模型的性能。
    # 其它权重参数需要权重衰减。
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # 生成max_new_tokens次
        for _ in range(max_new_tokens):
            # 如果目前序列的长度超过了block size，就切割后block size大小的序列来生成，用滑动窗口的方式控制长度
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 经过forward函数，得到logits，此时logits的shape是(batch_size, 1, vocab_size)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # temperature是预测概率的一个超参数
            # 加入temperature来处理logits，temperature越大，越容易sample到低概率的token
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                # v的大小是(batch_size, min(top_l, logits.size(-1)))，保存的是从大到小的topk个值
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 将小于topk的都赋值为-inf，使得它们在softmax之后为0
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # multinomial函数是用probs作为采样概率，采样num_samples个，返回下标，在这里就是根据概率对所有batch采样出来
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # 把生成出来的这个加到序列后面，继续生成
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
