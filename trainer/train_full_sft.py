import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import time  # 时间统计
import warnings  # 警告控制
import torch  # PyTorch框架
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器
from model.MokioModel import MokioMindConfig  # 模型配置
from dataset.lm_dataset import SFTDataset  # 监督微调数据集
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)  # 训练工具函数

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个epoch的主核心函数

    Args:
        epoch: 当前epoch序号
        loader: 数据加载器
        iters: 该epoch总迭代次数
        start_step: 起始步数（用于断点续训）
        wandb: 实验跟踪系统
    """
    start_time = time.time()  # 记录开始时间

    # 遍历所有数据批次
    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        # 📚 SFT特有：直接从数据集获取input_ids、labels和attention_mask
        # 与Pretrain不同，Pretrain需要(X, Y, loss_mask)三元组和手动计算loss

        # 将数据移到指定设备（GPU/CPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(
            args.device
        )  # ！修正：接收并转移attention_mask

        # 📚 学习率调度：使用余弦退火+预热策略
        # 从初始学习率逐渐降低到接近0
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 📚 混合精度前向传播：在autocast上下文中执行
        # 关键运算保持float32精度，其他运算用float16/bfloat16
        with autocast_ctx:
            # 📚 SFT特有：模型直接返回loss
            # 调用model(input_ids, labels=labels, attention_mask=attention_mask)触发损失函数计算
            # Pretrain则是调用model(X)只获取logits，需要手动计算loss
            res = model(
                input_ids, labels=labels, attention_mask=attention_mask
            )  # ！修正：加入attention_mask

            # SFT总损失 = 主任务loss + 辅助loss（MoE路由辅助）
            loss = res.loss + res.aux_loss

            # 📚 梯度累积：将loss平均化
            # 在多个step后才进行参数更新，模拟更大的batch_size
            loss = loss / args.accumulation_steps

        # 📚 amp.GradScaler：混合精度梯度缩放
        # 因为float16精度有效范围小，需要缩放梯度避免下溢
        scaler.scale(loss).backward()

        # 📚 梯度累积达到阈值，执行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 还原梯度的真实值（从缩放状态恢复）
            scaler.unscale_(optimizer)

            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行参数更新
            scaler.step(optimizer)
            # 更新GradScaler的缩放因子
            scaler.update()

            # 清空梯度，为下一次积累做准备
            optimizer.zero_grad(set_to_none=True)

        # 📚 日志记录：定期输出训练指标
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 恢复真实的loss值（乘回accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            # 获取辅助loss（如果存在）
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 主任务loss = 总loss - 辅助loss
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            # 计算剩余时间（单位：分钟）
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        # 📚 模型检查点保存：定期保存训练状态
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式（禁用dropout等）

            # 构建保存路径（根据是否使用MoE添加后缀）
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            # 📚 分布式模型处理：DDP模型需要通过.module访问真实模型
            # 其他情况下使用torch.compile的_orig_mod
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()

            # 📚 半精度保存：将float32参数转为float16节省存储空间
            # 模型权重保存为半精度可以减小文件大小（约50%）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 保存完整训练状态（包括优化器、epoch、step等）
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scaler=scaler,
            )

            model.train()  # 恢复训练模式
            del state_dict  # 释放内存

        # 释放显存，加快垃圾回收
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MokioMind Full SFT")

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="full_sft", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # ========== 训练策略参数 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len",
        default=340,
        type=int,
        help="训练的最大截断长度（中文1token≈1.5~1.7字符）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== 数据和恢复参数 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/sft_mini_512.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="pretrain",
        type=str,
        help="基于哪个权重训练，为none则不基于任何权重训练",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-Full-SFT", help="wandb项目名"
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用torch.compile加速（0=否，1=是）",
    )

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    """
    📚 分布式训练初始化知识点：
    - local_rank: 当前进程在本机上的GPU编号
    - 随机种子: 确保不同进程有不同但可复现的随机序列
    """
    local_rank = init_distributed_mode()  # 初始化分布式环境
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 分布式训练时使用对应GPU
    setup_seed(
        42 + (dist.get_rank() if dist.is_initialized() else 0)
    )  # 不同进程使用不同种子

    # ========== 2. 配置目录、模型参数、检查点 ==========
    """
    📚 SFT特有：基于预训练模型微调
    - 通常from_weight='pretrain'，表示加载预训练权重
    - Pretrain脚本中from_weight='none'表示从头开始
    """
    os.makedirs(args.save_dir, exist_ok=True)  # 确保保存目录存在
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    # 尝试加载断点续训数据
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    """
    📚 混合精度训练知识点：
    - bfloat16: Google开发，数值范围大，更稳定，推荐使用
    - float16: 标准半精度，节省内存但可能溢出
    - autocast: 自动选择精度，关键运算用float32
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU不支持autocast，使用nullcontext作为空操作
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置实验跟踪系统 ==========
    """
    📚 实验跟踪系统知识点：
    - SwanLab: 国产替代WandB的方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None  # 必须恢复到同一实验
        wandb_run_name = f"MokioMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    """
    📚 SFT vs Pretrain 数据集差异：
    - SFT: SFTDataset - 监督微调数据集，包含instruction和response
    - Pretrain: PretrainDataset - 预训练数据集，包含原始文本和mask
    """
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 📚 torch.compile加速：JIT编译模型获得20%~40%性能提升
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    # 加载SFT数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器：确保不同进程训练不同数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 混合精度梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    # AdamW优化器：包含权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从检查点恢复训练状态 ==========
    """
    📚 断点续训恢复：
    - 模型参数状态
    - 优化器状态（动量、方差估计等）
    - 梯度缩放器状态
    - 训练进度（epoch和step）
    """
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. DDP包装模型 ==========
    """
    📚 DistributedDataParallel特殊处理：
    - freqs_cos, freqs_sin是RoPE位置编码缓存，不需要梯度同步
    - 这样可以避免不必要的通信开销
    """
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch，确保数据随机打乱
        train_sampler and train_sampler.set_epoch(epoch)

        # 📚 随机数据排列：为每个epoch产生不同的数据顺序
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()

        # 📚 断点续训处理：
        # 第一个epoch且有检查点时，跳过已训练的step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
