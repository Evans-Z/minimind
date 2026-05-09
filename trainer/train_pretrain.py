import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from model.model_minimind_mhc import MiniMindMHCConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, tb_writer=None):
    start_time = time.time()
    last_step = start_step
    last_grad_norm = None
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            last_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            if tb_writer and is_main_process():
                global_step = epoch * iters + step
                tb_writer.add_scalar("train/loss", current_loss, global_step)
                tb_writer.add_scalar("train/logits_loss", current_logits_loss, global_step)
                tb_writer.add_scalar("train/aux_loss", current_aux_loss, global_step)
                tb_writer.add_scalar("train/lr", current_lr, global_step)
                if last_grad_norm is not None:
                    tb_writer.add_scalar("train/grad_norm", last_grad_norm, global_step)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            variant_suffix = f'_{args.model_variant}' if args.model_variant else ''
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{variant_suffix}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                model_variant=args.model_variant,
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        last_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "mps", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--model_variant', default='minimind', type=str, choices=['minimind', 'mhc'], help="选择模型实现：minimind或mhc")
    parser.add_argument('--hc_mult', default=4, type=int, help="mHC并行残差流数量")
    parser.add_argument('--hc_iters', default=20, type=int, help="mHC Sinkhorn迭代次数")
    parser.add_argument('--hc_eps', default=1e-6, type=float, help="mHC数值稳定项")
    parser.add_argument(
        '--hc_projector',
        default='sinkhorn',
        type=str,
        choices=['sinkhorn', 'balm'],
        help="mHC流混合矩阵投影器：sinkhorn或balm",
    )
    parser.add_argument('--hc_balm_r', default=1.0, type=float, help="BALM投影的惩罚系数r")
    parser.add_argument('--hc_balm_delta', default=1e-6, type=float, help="BALM投影的数值稳定项delta")
    parser.add_argument('--hc_balm_diag_cost', default=0.0, type=float, help="BALM线性代价系数lambda，对角代价矩阵C=lambda*I")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_tensorboard", action="store_true", help="是否使用TensorBoard")
    parser.add_argument("--tensorboard_logdir", type=str, default="../runs/pretrain", help="TensorBoard日志目录")
    parser.add_argument("--tb_run_tag", type=str, default="", help="TensorBoard运行标签，如baseline/use_mhc")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_variant == 'mhc':
        lm_config = MiniMindMHCConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            hc_mult=args.hc_mult,
            hc_iters=args.hc_iters,
            hc_eps=args.hc_eps,
            hc_projector=args.hc_projector,
            hc_balm_r=args.hc_balm_r,
            hc_balm_delta=args.hc_balm_delta,
            hc_balm_diag_cost=args.hc_balm_diag_cost,
        )
    else:
        lm_config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        )
    ckp_data = (
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            save_dir='../checkpoints',
            model_variant=args.model_variant,
        )
        if args.from_resume == 1
        else None
    )
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "mps"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "mps" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 4.1 TensorBoard ==========
    tb_writer = None
    if args.use_tensorboard and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise RuntimeError("TensorBoard未安装，请先执行: pip install tensorboard") from e
        tb_run_tag = args.tb_run_tag.strip()
        safe_tag = tb_run_tag.replace(" ", "_").replace("/", "_")
        run_name = f"{args.save_weight}_h{args.hidden_size}_{time.strftime('%Y%m%d-%H%M%S')}"
        tb_log_dir = os.path.join(args.tensorboard_logdir, safe_tag, run_name) if safe_tag else os.path.join(args.tensorboard_logdir, run_name)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        tb_writer.add_text("meta/run_tag", tb_run_tag if tb_run_tag else "none", 0)
        Logger(f"TensorBoard日志目录: {tb_log_dir}, tag: {tb_run_tag if tb_run_tag else 'none'}")
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, model_variant=args.model_variant)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, tb_writer)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, tb_writer)

    if tb_writer:
        tb_writer.close()
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()