"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# bfloat16 比 float16 更加紧凑，但是在训练的时候会有一些精度损失，所以在训练的时候需要小心使用。
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# 分布式训练的相关代码，用到了DDP来实现
# STEP 1：这里用到了RANK和LOCAL_RANK这两个环境变量，在DDP中，会给多进程中的每个进程分配独特的rank和local rank值。
# rank表示当前进程在分布式集群中的进程编号（就是说不是系统的pid，而是对当前这个程序的所有进程编号），
# 而local_rank表示当前进程在当前机器上的编号。（这里提一下环境变量，每个进程有自己独立的环境变量，在创建的时候都继承了全局环境变量和父进程环境变量）
# 这样设置rank和local rank的目的是为了让每个进程能够知道自己在分布式集群中的位置，方便在分布式训练中进行通信和同步。
# various inits, derived attributes, I/O setup
# 得到RANK环境变量的值，如果没有就是-1，说明没有使用分布式训练
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # 初始化分布式环境，init_process_group接收一个可选的参数backend，这里是nccl
    # 这个函数会在所有的进程中调用，它会启动一个主进程来管理整个进程组。
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # 根据local rank这个在当前设备上的编号确定该用哪个GPU
    device = f'cuda:{ddp_local_rank}'
    # 判断是不是主进程，主进程会执行logging和checkpointing等操作
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    # 每个进程设置不同的seed来保证训练的随机性
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    # 在分布式训练中，按比例减少每个进程的梯度累积迭代次数，以便在所有进程中均匀分配计算负载。
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    # 如果不是分布式训练，那么当前进程就是主进程，seed_offset设置为0.
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
# 允许在矩阵乘法（matmul）上使用 Tensor Core（tf32）
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
# 允许在 CUDA 动态神经网络库（CuDNN）上使用 Tensor Core（tf32）
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
# 根据 dtype 参数的值设置 PyTorch 的数据类型。
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 如果程序正在运行在 CPU 上，那么使用 nullcontext()；如果程序正在运行在 GPU 上，那么使用 torch.amp.autocast 函数，并设置相应的设备类型和数据类型。
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# nullcontext() 是 PyTorch 的一个函数，用于在 CPU 上运行程序时返回一个空的上下文。这样做的目的是为了避免在 CPU 上使用 autocast 函数导致的额外计算负担。
# torch.amp.autocast 函数是 PyTorch 的一个自动混合精度计算函数。它可以在运行时自动地切换数据类型，以便在需要时使用高精度，并在不需要时使用低精度。这可以提高程序的运行效率。

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
# 新建一个模型
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

# 从checkpoint中恢复模型
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # 判断一下checkpoint里存的和我们现在这个是不是匹配的
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # 这段代码是在修复checkpoint中的state_dict的key。
    # 在某些情况下，state_dict的key会带有一个"_orig_mod."的前缀，
    # 这段代码就是在遍历state_dict的所有键值对
    # 如果某个键的前缀是 unwanted_prefix，那么就将这个键值对从 state_dict 中移除，并添加一个新的键值对，新的键是原键去掉 unwanted_prefix 后的部分，值不变。
    # state_dict.pop(k) 会从 state_dict 中移除键为 k 的项，并返回其值。
    # state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k) 就是在 state_dict 中添加一个新的键值对，新的键是 k[len(unwanted_prefix):]，值是 state_dict.pop(k)。

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# 用openAI的weight
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# 之后是对模型进行编译，compile是PyTorch 2.0中新增加的一个功能，它可以将模型编译成一种新的形式，以提高运行速度。
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# STEP 2：把model放到DDP容器里去，这样就可以在多个GPU上进行训练了
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
# @torch.no_grad() 装饰器声明这个函数不需要计算梯度，这样可以节省计算资源，因为在评估模型时通常不需要计算梯度。
@torch.no_grad()
def estimate_loss():
    out = {}
    # 模型设置为评估模式，这是因为在评估模型时，我们不需要使用到如 Dropout 等只在训练模式下才会使用的层。
    model.eval()
    for split in ['train', 'val']:
        # 创建一个全零的张量 losses，用于存储每个批次的损失
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            # 将计算得到的损失添加到 losses 张量中。最后，计算 losses 张量的均值，并将结果存储到 out 字典中。
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 这段代码实现学习率的变化
# 余弦衰减是一种学习率调整策略，它的基本思路是在训练的开始阶段使用较大的学习率，然后在训练的后期降低学习率。
# 具体来说，它在训练过程中会将学习率按照一个余弦函数进行衰减，在训练开始时学习率较大，在训练后期逐渐降低到最小值。
# 这样做的好处是能够在训练开始时较快地接近最优解，并且在后期能够防止过拟合。
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) 在warmup_iters步内使用线性增长，即使学习率每步增加learning_rate * iter / warmup_iters
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) 当iter> lr_decay_iters时，返回最小学习率min_lr。
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) 在这之间，使用余弦衰减，最终值为最小学习率min_lr。
    # 3) in between, use cosine decay down to min learning rate
    # 用余弦衰减策略计算学习率。首先，计算衰减比例 decay_ratio，然后计算余弦衰减系数 coeff，最后根据 min_lr、coeff 和 learning_rate 计算学习率。
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



# 下面这一块是log相关的部分，代码中用到的工具是wandb，一个类似tensorboard的可视化工具，使用的方法就是用init初始化project，把需要记录的log用log的函数记录。
# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
# STEP 3：在训练的时候如果使用了ddp，现在model是一个container，里面的module才是我们的模型，所以这里需要unwrap一下
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # 判断是否需要进行学习率衰减。如果需要，就调用 get_lr 函数
    # 来计算当前迭代次数对应的学习率，并将这个学习率赋值给 optimizer 的 param_group。
    # 如果不需要，就直接使用预设的学习率。
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 如果到了需要eval的时候并且是master process，就计算一下现在的loss，根据eval的值来保存checkpoint
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 可以通过对 gradient_accumulation_steps 的设置模拟更大 batch size
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        # STEP4：在训练中，只需要在最后一个微步中同步梯度。官方的做法是使用model.no_sync()上下文管理器，
        # 但是这段代码直接设置了model.require_backward_grad_sync变量，当micro_step等于gradient_accumulation_steps - 1时，需要同步梯度。
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    # 在训练神经网络时对梯度进行裁剪，以防止梯度爆炸的问题。
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    # 代码调用 optimizer.zero_grad(set_to_none=True) 来清零优化器中的梯度。
    # 这是因为在 PyTorch 中，梯度是累积的，如果不清零，那么每次计算梯度时，新的梯度会被加到旧的梯度上，导致结果错误。
    # 参数 set_to_none=True 表示将梯度张量的数据直接设为 None，这样可以更早地释放梯度占用的内存。
    optimizer.zero_grad(set_to_none=True)

    # 计算时间并且记录日志
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        # loss.item() 来获取损失值，并乘以 gradient_accumulation_steps 来得到总的损失值。
        # 这是因为在计算损失时，可能会对损失进行平均，所以在记录损失时，需要将其缩放回原来的总损失。
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            # estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS。
            # 然后，使用一个滑动平均的方式来更新 running_mfu。
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 超过限定次数就退出
    # termination conditions
    if iter_num > max_iters:
        break

# STEP 5：最后，调用了destroy_process_group()来销毁进程组
# 注意，如果是当前的process是master_process，还需要执行创建output dir，初始化wandb，记录log，计算loss，保存checkpoint。
if ddp:
    destroy_process_group()

