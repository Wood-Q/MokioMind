import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 命令行参数解析
import re        # 正则表达式，用于奖励计算
import warnings  # 警告控制
import torch     # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
import torch.nn.functional as F   # 神经网络函数
from transformers import AutoTokenizer  # HuggingFace分词器
from contextlib import nullcontext  # 上下文管理器
from torch import optim, nn  # 优化器和神经网络
from torch.nn.parallel import DistributedDataParallel  # 分布式并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度
from transformers import AutoModel  # HuggingFace模型加载
from model.MokioModel import MokioMindConfig, MokioMindForCausalLM  # MiniMind模型
from dataset.lm_dataset import RLAIFDataset  # RL数据集
from trainer.trainer_utils import (  # 训练工具函数
    Logger, is_main_process, lm_checkpoint, init_distributed_mode, 
    setup_seed, SkipBatchSampler, init_model
)

warnings.filterwarnings('ignore')
#==========Critic Model部分==========

class CriticModel(MokioMindForCausalLM):
    def __init__(self,params):
        super().__init__(params)
        # 价值头，用于输出每个token位置的状态价值
        self.value_head=nn.Linear(params.hidden_size,1)
        
    def forward(self,input_ids=None,attention_mask=None,**kwargs):
        outputs=self.model(input_ids=input_ids,attention_mask=attention_mask,**kwargs)
        hidden_states=self.model.norm(outputs[0])
        
        values=self.value_head(hidden_states).squeeze(-1)
        return values

#==========奖励计算部分==========
def calculate_rewards(prompts,responses,reward_model,reward_tokenizer):
    def reasoning_model_reward(rewards):
        # 使用正则表达式匹配思考-回答格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # 多了一个\n，考虑到think和answer之间有空行的情况
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        # 通过正则表达式计算奖励，如果回答符合格式则奖励0.5，否则0.0
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]
        
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)
        
        def mark_num(text):
            reward=0
            if text.count("<think>")==1:
                reward+=0.25
            if text.count("</think>")==1:
                reward+=0.25
            if text.count("<answer>")==1:
                reward+=0.25
            if text.count("</answer>")==1:
                reward+=0.25
            return reward
        
        mark_rewards=[mark_num(response) for response in responses]
        rewards+=torch.tensor(mark_rewards,device=args.device)
        return rewards
    rewards=torch.zeros(len(responses),device=args.device)
    
    if args.reasoning==1:
        rewards=reasoning_model_reward(rewards)
#==========Reward模型评分部分==========
    with torch.no_grad():
        reward_model_scores = []
        for prompt,response in zip(prompts,responses):
            
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            
            tmp_chat=messages+[{"role":"assistant","content":response}]
            score=reward_model.get_reward(tmp_chat,reward_tokenizer)
            
            scale=3.0
            score=max(min(score,scale),-scale)
            
            if args.reasoning==1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 📚 加权组合
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)
        
        reward_model_scores=torch.tensor(reward_model_scores,device=args.device)
        rewards+=reward_model_scores
        
    return rewards

#==========PPO训练一个Epoch部分==========
def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    # 切换actor和critic模型到训练模式
    actor_model.train()
    critic_model.train()
    
    for step,batch in enumerate(loader,start=start_step+1):
        prompts=batch['prompt']
        # 编码输入
        enc=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=args.max_seq_len).to(args.device)
        # 计算每个prompt的长度（用于后续处理）
        prompt_lengths=enc.attention_mask.sum(dim=1)
        
        with torch.no_grad():
            model_for_gen=actor_model.module if isinstance(actor_model,DistributedDataParallel) else actor_model
            
            gen_out=model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码生成的响应
        responses_text=[tokenizer.decode(gen_out[i,prompt_lengths[i]:],skip_special_tokens=True) for i in range(len(prompts))]
        
        # 计算奖励
        rewards=calculate_rewards(prompts,responses_text,reward_model,reward_tokenizer)
        
        # 创建一个mask，用于标记哪些位置上是有效token
        full_mask=(gen_out!=tokenizer.pad_token_id).long()
        # critic模型进行价值估计
        value_seq=critic_model(input_ids=gen_out,attention_mask=full_mask)
        # 拿到最后一个非pad位置的索引
        last_indices=full_mask.sum(dim=1)-1
        # 获取每条序列最后token的value
        values=value_seq[torch.arange(len(last_indices)),last_indices]
        # advantage=reward-估计的value
        advantages = rewards - values.detach()  # [B]
        
        # 计算actor log，表示actor对这个答案的“信心”
        # 先生成logits
        logits=actor_model(input_ids=gen_out,attention_mask=full_mask).logits  # [B, L, V]
        # label是生成的token序列，去掉第一个token（因为logits是预测下一个token的概率）
        labels=gen_out[:,1:].clone()
        # 使用log_softmax计算log概率
        logp_tokens=F.log_softmax(logits[:,:-1,:],dim=-1).gather(2,labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        seq_len=gen_out.size(1)-1
        # 只关心response部分的概率，所以要把prompts部分的mask掉
        resp_mask=torch.arange(seq_len,device=gen_out.device).unsqueeze(0)>=prompt_lengths.unsqueeze(1)
        
        final_mask=resp_mask&(~labels.eq(tokenizer.pad_token_id))
        # 把所有response部分的log概率加起来，得到每条序列的总log概率
        actor_logp=(logp_tokens*final_mask).sum(dim=1)

        # 计算old和ref log的概率
        # old用于防止策略更新过大，ref用于计算KL惩罚，防止模型忘本
        with torch.no_grad():
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]
            
        # 计算KL散度和ratio
        kl=(actor_logp - old_logp).mean()
        kl_ref=(actor_logp - ref_logp).mean()
        ratio=torch.exp(actor_logp - old_logp)  # [B]
        
        # PPO裁剪损失
        surr1=ratio*advantages  # [B]
        surr2=torch.clamp(ratio,1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)*advantages  # [B]
        policy_loss=-torch.min(surr1,surr2).mean()
        
        # 价值函数损失
        value_loss=F.mse_loss(values,rewards)
        # 总损失
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        loss.backward()
        
        # 更新参数
        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
        # 📚 日志记录
        if is_main_process():
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")

        # 📚 更新old actor
        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # 📚 模型保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            


if __name__ == "__main__":
    """
    PPO主函数：近端策略优化脚本的入口点
    
    📚 PPO训练架构：
    1. Actor模型：生成策略，输出动作概率
    2. Critic模型：价值函数，估计状态价值
    3. Reward模型：奖励函数，评估生成质量
    4. Old Actor：用于重要性采样的旧策略
    5. Reference：用于KL惩罚的参考策略
    """
    
    # 📚 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    
    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size（PPO batch较小）")
    
    # 📚 PPO学习率设置
    # PPO学习率通常很小，避免策略剧烈变化
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    
    # ========== 硬件配置 ==========
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    
    # ========== 训练策略 ==========
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    
    # ========== 模型架构参数 ==========
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # ========== PPO生成参数 ==========
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    
    # ========== 数据和模型参数 ==========
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    
    # 📚 PPO超参数
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数（控制策略更新幅度）")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    
    # 📚 推理模型配置
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    
    # 📚 Reward模型路径
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    
    args = parser.parse_args()
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    # ========== 5. 初始化模型和数据 ==========
    # 📚 PPO模型架构
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # 📚 Actor模型（策略模型）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = 'left'  # PPO需要左侧padding
    
    # 📚 Old Actor模型（用于重要性采样）
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    
    # 📚 Reference模型（用于KL惩罚）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # 📚 Critic模型（价值函数）
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    
    # 📚 Reward模型（奖励函数）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # 📚 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包装模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)