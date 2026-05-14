import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from model.model_minimind_mhc import MiniMindMHCConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    setup_seed,
)

try:
    import yaml

    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

try:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh

    _HAS_FSDP2 = True
except Exception:
    fully_shard = None
    StateDictOptions = None
    get_state_dict = None
    set_state_dict = None
    init_device_mesh = None
    _HAS_FSDP2 = False

warnings.filterwarnings("ignore")

args = None
model = None
optimizer = None
scaler = None
autocast_ctx = None
lm_config = None
save_paths = {}
run_is_fsdp2 = False
model_config_overrides = {}


def _apply_fsdp2_sharding(root_model: torch.nn.Module):
    # Compose bottom-up sharding where we have natural block boundaries.
    core = getattr(root_model, "model", None)
    if core is not None and hasattr(core, "layers"):
        for layer in core.layers:
            fully_shard(layer, reshard_after_forward=bool(args.fsdp2_reshard_after_forward))
        fully_shard(core, reshard_after_forward=bool(args.fsdp2_reshard_after_forward))
    fully_shard(root_model, reshard_after_forward=bool(args.fsdp2_reshard_after_forward))


def _checkpoint_paths():
    variant_suffix = f"_{args.model_variant}" if args.model_variant else ""
    moe_suffix = "_moe" if lm_config.use_moe else ""
    stem = f"{args.tagged_save_weight}_{lm_config.hidden_size}{variant_suffix}{moe_suffix}"
    os.makedirs(args.save_dir, exist_ok=True)
    return {
        "weight": os.path.join(args.save_dir, f"{stem}.pth"),
        "resume": os.path.join(args.save_dir, f"{stem}_resume.pth"),
    }


def _coerce_override_value(key: str, value: Any) -> Any:
    if key in {"use_moe", "hc_balm_trainable_r"}:
        return int(bool(value))
    return value


def _apply_single_preset(preset_name: str, preset_table: dict[str, Any], label: str, source_path: str):
    if preset_name not in preset_table:
        available = ", ".join(sorted(preset_table.keys()))
        raise ValueError(f"{label} preset '{preset_name}' not found. Available: {available}")
    preset = preset_table[preset_name]
    if not isinstance(preset, dict):
        raise ValueError(f"{label} preset '{preset_name}' must be a mapping")

    for key, value in preset.items():
        value = _coerce_override_value(key, value)
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            model_config_overrides[key] = value
    Logger(f"Applied {label} preset '{preset_name}' from {source_path}")


def _apply_model_preset_overrides():
    global model_config_overrides
    model_config_overrides = {}
    if not args.model_config_yaml:
        return
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is required for --model_config_yaml. Please install via: pip install pyyaml")

    with open(args.model_config_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if args.size_preset:
        size_presets = data.get("size_presets")
        if not isinstance(size_presets, dict):
            raise ValueError("YAML must contain top-level 'size_presets' for --size_preset")
        _apply_single_preset(args.size_preset, size_presets, "size", args.model_config_yaml)
    if args.context_preset:
        context_presets = data.get("context_presets")
        if not isinstance(context_presets, dict):
            raise ValueError("YAML must contain top-level 'context_presets' for --context_preset")
        _apply_single_preset(args.context_preset, context_presets, "context", args.model_config_yaml)

    Logger(
        f"Resolved preset config: hidden_size={args.hidden_size}, layers={args.num_hidden_layers}, "
        f"use_moe={args.use_moe}, max_seq_len={args.max_seq_len}"
    )


def _save_checkpoint(epoch: int, step: int):
    if not is_main_process():
        return
    raw_model = model
    model_state = None
    optim_state = None

    if run_is_fsdp2:
        state_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        model_state, optim_state = get_state_dict(
            raw_model,
            optimizer,
            options=state_options,
        )
    else:
        if isinstance(raw_model, DistributedDataParallel):
            raw_model = raw_model.module
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        model_state = raw_model.state_dict()
        optim_state = optimizer.state_dict()

    weight_tmp = save_paths["weight"] + ".tmp"
    resume_tmp = save_paths["resume"] + ".tmp"
    torch.save({k: v.half().cpu() for k, v in model_state.items()}, weight_tmp)
    os.replace(weight_tmp, save_paths["weight"])

    resume_data = {
        "model": model_state,
        "optimizer": optim_state,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "dist_backend": args.dist_backend,
    }
    torch.save(resume_data, resume_tmp)
    os.replace(resume_tmp, save_paths["resume"])


def _load_resume():
    if not args.from_resume:
        return 0, 0
    if not os.path.exists(save_paths["resume"]):
        Logger(f"resume checkpoint not found: {save_paths['resume']}")
        return 0, 0

    ckp_data = torch.load(save_paths["resume"], map_location="cpu")
    saved_backend = ckp_data.get("dist_backend", "ddp")
    if saved_backend != args.dist_backend:
        Logger(f"Warning: checkpoint backend={saved_backend}, current backend={args.dist_backend}")

    if run_is_fsdp2:
        set_state_dict(
            model,
            optimizer,
            model_state_dict=ckp_data["model"],
            optim_state_dict=ckp_data.get("optimizer"),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        model.load_state_dict(ckp_data["model"], strict=False)
        if "optimizer" in ckp_data:
            optimizer.load_state_dict(ckp_data["optimizer"])

    if "scaler" in ckp_data:
        scaler.load_state_dict(ckp_data["scaler"])
    return ckp_data.get("epoch", 0), ckp_data.get("step", 0)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, tb_writer=None):
    start_time = time.time()
    last_step = start_step
    last_grad_norm = None
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, "
                f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
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
            if tb_writer and is_main_process():
                global_step = epoch * iters + step
                tb_writer.add_scalar("train/loss", current_loss, global_step)
                tb_writer.add_scalar("train/logits_loss", current_logits_loss, global_step)
                tb_writer.add_scalar("train/aux_loss", current_aux_loss, global_step)
                tb_writer.add_scalar("train/lr", current_lr, global_step)
                if last_grad_norm is not None:
                    tb_writer.add_scalar("train/grad_norm", last_grad_norm, global_step)

        if step % args.save_interval == 0 or step == iters:
            model.eval()
            _save_checkpoint(epoch=epoch, step=step)
            model.train()

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        _ = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def _build_config():
    config_kwargs = dict(model_config_overrides)
    # Remove explicit constructor args from override kwargs to avoid duplicate kwargs.
    config_kwargs.pop("hidden_size", None)
    config_kwargs.pop("num_hidden_layers", None)
    config_kwargs.pop("use_moe", None)
    if args.model_variant == "mhc":
        config_kwargs.pop("hc_mult", None)
        config_kwargs.pop("hc_iters", None)
        config_kwargs.pop("hc_eps", None)
        config_kwargs.pop("hc_projector", None)
        config_kwargs.pop("hc_balm_r", None)
        config_kwargs.pop("hc_balm_trainable_r", None)
        config_kwargs.pop("hc_balm_delta", None)
        config_kwargs.pop("hc_balm_diag_cost", None)
        config_kwargs.pop("hc_balm_offdiag_cost", None)
        config_kwargs.pop("hc_balm_cost_scale", None)
        return MiniMindMHCConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            hc_mult=args.hc_mult,
            hc_iters=args.hc_iters,
            hc_eps=args.hc_eps,
            hc_projector=args.hc_projector,
            hc_balm_r=args.hc_balm_r,
            hc_balm_trainable_r=bool(args.hc_balm_trainable_r),
            hc_balm_delta=args.hc_balm_delta,
            hc_balm_diag_cost=args.hc_balm_diag_cost,
            hc_balm_offdiag_cost=args.hc_balm_offdiag_cost,
            hc_balm_cost_scale=args.hc_balm_cost_scale,
            **config_kwargs,
        )
    return MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        **config_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Scale Pretraining (DDP/FSDP2)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="pretrain_scale", type=str, help="保存权重的前缀名")
    parser.add_argument(
        "--model_config_yaml",
        type=str,
        default="",
        help="模型结构YAML配置文件路径（用于固定不同规模如64M/1B/2B）",
    )
    parser.add_argument(
        "--size_preset",
        type=str,
        default="",
        help="YAML中的规模preset名称（来自size_presets）",
    )
    parser.add_argument(
        "--context_preset",
        type=str,
        default="",
        help="YAML中的上下文preset名称（来自context_presets）",
    )
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
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=340, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--model_variant", default="minimind", type=str, choices=["minimind", "mhc"], help="选择模型实现")
    parser.add_argument("--hc_mult", default=4, type=int, help="mHC并行残差流数量")
    parser.add_argument("--hc_iters", default=20, type=int, help="mHC Sinkhorn迭代次数")
    parser.add_argument("--hc_eps", default=1e-6, type=float, help="mHC数值稳定项")
    parser.add_argument("--hc_projector", default="sinkhorn", type=str, choices=["sinkhorn", "balm"], help="mHC投影器")
    parser.add_argument("--hc_balm_r", default=1.0, type=float, help="BALM投影惩罚系数r")
    parser.add_argument("--hc_balm_trainable_r", default=0, type=int, choices=[0, 1], help="是否将BALM r设为可训练")
    parser.add_argument("--hc_balm_delta", default=1e-6, type=float, help="BALM投影稳定项delta")
    parser.add_argument("--hc_balm_diag_cost", default=0.0, type=float, help="BALM对角代价")
    parser.add_argument("--hc_balm_offdiag_cost", default=0.0, type=float, help="BALM非对角代价")
    parser.add_argument("--hc_balm_cost_scale", default=1.0, type=float, help="BALM代价缩放")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测并续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_tensorboard", action="store_true", help="是否使用TensorBoard")
    parser.add_argument("--tensorboard_logdir", type=str, default="../runs/pretrain_scale", help="TensorBoard日志目录")
    parser.add_argument("--tb_run_tag", type=str, default="", help="TensorBoard运行标签")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--dist_backend", default="fsdp2", type=str, choices=["ddp", "fsdp2"], help="分布式后端")
    parser.add_argument(
        "--fsdp2_reshard_after_forward",
        default=1,
        type=int,
        choices=[0, 1],
        help="FSDP2是否在forward后重分片",
    )
    args = parser.parse_args()

    tb_run_tag = args.tb_run_tag.strip()
    safe_tag = tb_run_tag.replace(" ", "_").replace("/", "_")
    args.run_tag_suffix = f"_{safe_tag}" if safe_tag else ""
    args.tagged_save_weight = f"{args.save_weight}{args.run_tag_suffix}"
    using_any_preset = bool(args.size_preset) or bool(args.context_preset)
    if using_any_preset and not args.model_config_yaml:
        raise ValueError("When using presets, --model_config_yaml is required")
    if args.model_config_yaml and not using_any_preset:
        raise ValueError(
            "With --model_config_yaml, provide at least one of: --size_preset, --context_preset"
        )
    _apply_model_preset_overrides()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    lm_config = _build_config()
    save_paths = _checkpoint_paths()

    device_type = "cuda" if "cuda" in args.device else "mps"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "mps" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_run_name = (
            f"MiniMind-PretrainScale-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    tb_writer = None
    if args.use_tensorboard and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise RuntimeError("TensorBoard未安装，请先执行: pip install tensorboard") from e
        run_name = f"{args.save_weight}_h{args.hidden_size}_{time.strftime('%Y%m%d-%H%M%S')}"
        tb_log_dir = (
            os.path.join(args.tensorboard_logdir, safe_tag, run_name)
            if safe_tag
            else os.path.join(args.tensorboard_logdir, run_name)
        )
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        tb_writer.add_text("meta/run_tag", tb_run_tag if tb_run_tag else "none", 0)
        Logger(f"TensorBoard日志目录: {tb_log_dir}, tag: {tb_run_tag if tb_run_tag else 'none'}")

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, model_variant=args.model_variant)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    run_is_fsdp2 = args.dist_backend == "fsdp2" and dist.is_initialized()
    if args.dist_backend == "fsdp2" and not _HAS_FSDP2:
        raise RuntimeError("FSDP2 is unavailable in current PyTorch build.")
    if run_is_fsdp2:
        torch.cuda.set_device(torch.device(args.device))
        _ = init_device_mesh("cuda", (dist.get_world_size(),))
        _apply_fsdp2_sharding(model)
        Logger(f"FSDP2 enabled, reshard_after_forward={bool(args.fsdp2_reshard_after_forward)}")
    elif dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger("DDP enabled")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = _load_resume()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, tb_writer)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, tb_writer)

    if tb_writer:
        tb_writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
