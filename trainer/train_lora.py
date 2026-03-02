"""
MiniMind LoRA (Low-Rank Adaptation) 微调脚本

📚 LoRA 核心知识点：
- 什么是LoRA：一种参数高效微调方法，只训练少量新增参数
- 原理：在预训练模型的权重矩阵旁边添加低秩分解矩阵 ΔW = BA
  - 原始权重 W 保持冻结（requires_grad=False）
  - 新增两个小矩阵 A(d×r) 和 B(r×d)，其中 r<<d（秩远小于维度）
  - 前向计算：output = Wx + BAx
- 优势对比：
  - Full SFT：更新所有参数，效果好但需要大显存和长时间
  - LoRA：只更新1-5%的参数，显存需求小，训练快，适合资源受限场景
  - 多任务切换：可以保存多组LoRA权重，快速切换不同任务能力

📚 适用场景：
- 个性化定制：医疗、法律、金融等垂直领域适配
- 快速实验：尝试不同数据/超参时，LoRA训练速度快
- 资源受限：单卡或小显存环境
"""

import os
import sys

# 📚 Python模块系统
# __package__: 显式声明当前模块所属的包
# sys.path.append: 将项目根目录加入模块搜索路径，使得可以导入project内的模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 命令行参数解析
import time      # 时间统计
import warnings  # 警告控制
import torch     # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器（无操作占位符）
from torch import optim, nn         # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# MiniMind相关组件
from model.model_minimind import MiniMindConfig     # 模型配置
from dataset.lm_dataset import SFTDataset          # 监督微调数据集
from model.model_lora import save_lora, apply_lora # LoRA权重保存和应用
from trainer.trainer_utils import (                # 训练工具函数
    get_lr, Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行单个LoRA训练轮次
    
    📚 LoRA训练的特殊之处：
    1. 只有LoRA参数参与梯度计算和更新
    2. 原始模型权重保持冻结，节省显存和计算
    3. 训练流程与Full SFT相同，但参数量小得多
    
    Args:
        epoch: 当前训练轮次
        loader: 数据加载器
        iters: 总迭代次数
        lora_params: LoRA参数列表（只有这些参数会被更新）
        start_step: 起始步数（用于断点续训）
        wandb: 实验跟踪工具
    """
    # 📚 交叉熵损失函数
    # reduction='none': 保持每个位置的损失值，不自动求平均
    # 这样可以配合loss_mask进行精确的损失计算
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    # 📚 enumerate的start参数
    # start=start_step + 1: 从指定步数开始计数，用于断点续训时保持step编号连续
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 📚 张量设备迁移
        # .to(device): 将CPU上的张量移动到GPU，必须保证数据和模型在同一设备
        X = X.to(args.device)          # 输入序列
        Y = Y.to(args.device)          # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码
        
        # 📚 动态学习率调整
        # 使用余弦退火策略，学习率随训练进度平滑下降
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        
        # 📚 优化器参数组
        # optimizer.param_groups: 列表，每个元素是一个参数组字典
        # 通过修改'lr'键来动态调整学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 📚 混合精度训练上下文
        # autocast_ctx: 自动混合精度，关键运算用float32，其他用float16/bfloat16
        # 可以加速训练并节省显存，同时保持数值稳定性
        with autocast_ctx:
            # 模型前向传播
            res = model(X)
            
            # 📚 损失计算详解
            # 1. res.logits: [batch_size, seq_len, vocab_size]
            # 2. view(-1, size): 将张量展平，-1表示自动计算该维度
            # 3. Y.view(-1): 将目标序列展平为一维
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # [batch*seq, vocab]
                Y.view(-1)                                 # [batch*seq]
            ).view(Y.size())  # 恢复为 [batch_size, seq_len]
            
            # 📚 掩码损失计算
            # 只对有效位置（非padding）计算损失
            # .sum(): 求和所有有效位置的损失
            # / loss_mask.sum(): 除以有效位置数量，得到平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 📚 MoE辅助损失
            # 如果使用MoE（混合专家）架构，需要加上负载均衡损失
            # 确保不同专家被均匀使用
            loss += res.aux_loss
            
            # 📚 梯度累积
            # 将损失除以累积步数，实现梯度累积效果
            # 等价于使用更大的batch size，但显存占用更小
            loss = loss / args.accumulation_steps

        # 📚 混合精度反向传播
        # scaler.scale(loss): 放大损失值，防止float16下的梯度下溢
        # .backward(): 计算梯度，填充到各参数的.grad属性
        scaler.scale(loss).backward()

        # 📚 梯度累积和参数更新
        # 每accumulation_steps步才真正更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 📚 梯度反缩放
            # scaler.unscale_(optimizer): 将放大的梯度恢复到真实值
            # 必须在梯度裁剪之前调用
            scaler.unscale_(optimizer)
            
            # 📚 梯度裁剪
            # clip_grad_norm_: 将梯度的L2范数限制在指定阈值内
            # 防止梯度爆炸，稳定训练过程
            # 注意：这里只裁剪lora_params，因为其他参数已被冻结
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            # 📚 优化器步进
            # scaler.step(optimizer): 执行参数更新 param = param - lr * grad
            # scaler.update(): 更新scaler的缩放因子，用于下一次迭代
            scaler.step(optimizer)
            scaler.update()

            # 📚 梯度清零
            # set_to_none=True: 将梯度设为None而不是0
            # 优点：节省内存，性能更好
            optimizer.zero_grad(set_to_none=True)

        # 📚 训练日志记录
        # 每log_interval步或最后一步打印一次日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 📚 .item()方法
            # 将单元素张量转换为Python标量
            # 必须恢复梯度累积的缩放：乘以accumulation_steps
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']  # 获取当前学习率
            
            # 📚 ETA计算（预计剩余时间）
            # (已用时间 / 已完成步数) * 总步数 = 预计总时间
            # 预计总时间 - 已用时间 = 预计剩余时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 📚 f-string格式化
            # {var:.6f}: 保留6位小数
            # {var:.12f}: 保留12位小数（学习率通常很小）
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            # 记录到实验跟踪系统
            if wandb: 
                # 📚 wandb.log()
                # 记录标量指标到WandB/SwanLab平台
                # 可以在网页端实时查看训练曲线
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 📚 LoRA模型检查点保存
        # 每save_interval步或最后一步保存一次
        # is_main_process(): 只有主进程保存，避免多进程重复写入
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式
            
            # 📚 LoRA权重保存路径
            # 只保存LoRA的A和B矩阵，不保存整个模型
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            
            # 📚 save_lora函数
            # 从模型中提取所有包含'lora'的参数并保存
            # 文件大小通常只有Full SFT的1-5%
            save_lora(model, lora_save_path)
            
            # 📚 完整训练状态保存
            # 保存模型、优化器、scaler、训练进度等
            # 用于断点续训
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, 
                         optimizer=optimizer, scaler=scaler, epoch=epoch, 
                         step=step, wandb=wandb, save_dir='../checkpoints')
            
            model.train()  # 恢复训练模式


if __name__ == "__main__":
    # 📚 命令行参数解析
    # argparse: Python标准库，用于解析命令行参数
    # 提供默认值和帮助信息，便于用户配置训练参数
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    
    # 📚 模型保存相关参数
    # save_dir: 指定LoRA权重和检查点的保存目录
    # lora_name: LoRA权重的标识符，用于区分不同任务的LoRA适配器
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    
    # 📚 训练超参数
    # epochs: 训练的总轮数，控制模型训练的完整程度
    # batch_size: 每个批次的样本数量，影响显存使用和训练稳定性
    # learning_rate: 初始学习率，控制参数更新的步长
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    
    # 📚 设备和精度配置
    # device: 指定训练使用的设备（GPU/CPU）
    # dtype: 混合精度训练的数据类型，bfloat16更稳定，float16更高效
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    
    # 📚 数据加载和训练优化
    # num_workers: 数据加载的并行进程数，提高数据读取效率
    # accumulation_steps: 梯度累积步数，模拟更大的batch size
    # grad_clip: 梯度裁剪阈值，防止梯度爆炸
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # 📚 日志和保存配置
    # log_interval: 每多少步打印一次训练日志
    # save_interval: 每多少步保存一次模型检查点
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")
    
    # 📚 模型架构参数
    # hidden_size: 模型隐藏层维度，影响模型容量和计算复杂度
    # num_hidden_layers: Transformer层数，层数越多模型越深
    # max_seq_len: 训练时序列的最大长度，影响显存使用
    # use_moe: 是否使用Mixture of Experts架构，提高模型效率
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 📚 数据和权重配置
    # data_path: 训练数据的文件路径，通常是JSONL格式
    # from_weight: 基于哪个预训练权重进行LoRA微调
    # from_resume: 是否从检查点恢复训练，支持断点续训
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 📚 实验跟踪配置
    # use_wandb: 是否启用WandB/SwanLab进行实验跟踪
    # wandb_project: WandB项目的名称，用于组织实验
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 📚 分布式训练初始化
    # init_distributed_mode(): 初始化多GPU分布式训练环境
    # 如果使用多卡，会设置进程组和本地rank
    local_rank = init_distributed_mode()
    
    # 📚 设备分配
    # 在分布式训练中，每个进程使用不同的GPU
    # dist.get_rank(): 获取当前进程的全局rank
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    
    # 📚 随机种子设置
    # setup_seed(): 设置随机种子，确保训练的可复现性
    # 不同进程使用不同的种子，避免生成相同的数据
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 📚 创建保存目录
    # os.makedirs: 递归创建目录，如果已存在则忽略
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 📚 模型配置初始化
    # MiniMindConfig: 定义模型的超参数，如隐藏维度、层数、是否使用MoE
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    
    # 📚 检查点检测
    # lm_checkpoint(): 检查是否存在可用的检查点
    # 如果from_resume=1，则尝试加载之前的训练状态
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 📚 设备类型判断
    # 根据设备字符串判断是CPU还是GPU
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 📚 数据类型选择
    # bfloat16: 更好的数值稳定性，适合现代GPU
    # float16: 更高的性能，但可能有精度损失
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 📚 自动混合精度上下文
    # autocast: 自动选择合适的精度进行计算
    # CPU模式下使用nullcontext（无操作）
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb ==========
    # 📚 实验跟踪初始化
    # SwanLab: 类似WandB的实验管理工具
    # 支持实验重启和指标记录
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        
        # 📚 WandB运行ID
        # 从检查点恢复时使用相同的ID，保持实验连续性
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        
        # 📚 实验名称生成
        # 包含关键参数，便于识别不同的实验配置
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 📚 模型初始化
    # init_model(): 加载预训练模型和tokenizer
    # from_weight指定基础权重文件
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 📚 应用LoRA适配器
    # apply_lora(): 在模型中注入LoRA参数
    # 为指定的层添加A和B矩阵
    apply_lora(model)
    
    # 📚 参数统计
    # 计算总参数量和LoRA参数量
    # LoRA参数通常只占总参数的1-5%
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 📚 参数冻结策略
    # 冻结非LoRA参数，只训练LoRA适配器
    # 收集需要优化的LoRA参数列表
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # 📚 数据集准备
    # SFTDataset: 监督微调数据集类
    # 处理JSONL格式的数据，进行tokenization和截断
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 📚 数据采样器
    # DistributedSampler: 分布式训练的数据采样器
    # 确保不同进程采样不同的数据子集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 📚 梯度缩放器
    # GradScaler: 用于float16训练的梯度缩放
    # 防止梯度下溢，提高训练稳定性
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 📚 优化器配置
    # AdamW: 常用的优化器，带有权重衰减
    # 只优化LoRA参数，原始参数保持冻结
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    # 📚 训练状态恢复
    # 如果存在检查点，从中恢复模型、优化器、scaler状态
    # 支持断点续训，节省训练时间
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP包装模型 ==========
    # 📚 分布式数据并行
    # DistributedDataParallel: PyTorch的DDP实现
    # 将模型包装为分布式版本，支持多GPU训练
    # _ddp_params_and_buffers_to_ignore: 忽略不需要同步的缓冲区（如位置编码）
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    # 📚 训练循环
    # 遍历每个epoch，执行训练过程
    # 支持从检查点恢复，继续未完成的训练
    for epoch in range(start_epoch, args.epochs):
        # 📚 采样器epoch设置
        # set_epoch(): 确保分布式采样器的随机性
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            # 📚 跳过已完成的step
            # SkipBatchSampler: 自定义采样器，跳过前N个batch
            # 用于断点续训时从指定step开始
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, lora_params, start_step, wandb)
        else: # 默认从头开始
            # 📚 标准数据加载器
            # DataLoader: PyTorch的数据加载器
            # shuffle: 单GPU时随机打乱，多GPU时由sampler控制
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
