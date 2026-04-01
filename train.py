import os
import math
import json
import random
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
from tqdm import tqdm

from network.network import WaveLetFusion, flow_warp
from network.loss import FusionRegistrationLoss
from data.dataset import IRVIFusionRegistrationDataset


@dataclass
class TrainConfig:
    dataset_root: str = "./data/roadscene_train"
    output_dir: str = "./outputs/ablation"
    vis_dir: str = "./outputs/vis_ablation"

    ir_dir: str = "ir"
    vi_dir: str = "vi"
    ir_d_dir: str = "ir_d"
    vi_d_dir: str = "vi_d"
    ir_flow_dir: str = "ir_flows"
    vi_flow_dir: str = "vi_flows"
    ir_valid_dir: str = "ir_valid"
    vi_valid_dir: str = "vi_valid"

    skip_stage1: bool = False
    skip_stage2_fusion: bool = False

    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    val_ratio: float = 0.15

    seed: int = 3407
    deterministic: bool = True
    use_amp: bool = True
    max_grad_norm: float = 1.0

    epochs_stage1: int = 50
    epochs_stage2: int = 30
    stage2_warmup_epochs: int = 10

    s1_wavelet_warmup_epochs: int = 20
    s1_ll_start_ratio: float = 0.3
    s1_hf_start_ratio: float = 0.2

    stage1_lr: float = 5e-5
    stage2_lr: float = 5e-6
    stage2_shared_lr: float = 1e-6
    stage2_reg_lr: float = 1e-7
    weight_decay: float = 5e-4

    scheduler_type: str = "cosine"
    step_size: int = 30
    gamma: float = 0.5
    min_lr_ratio: float = 0.05

    s1_reg_epe: float = 0.5
    s1_reg_smooth: float = 0.25
    s1_reg_photo: float = 0.0
    s1_reg_edge: float = 0.0
    s1_reg_ll: float = 0.20
    s1_reg_hf: float = 0.10

    s2_fusion_grad: float = 5.0
    s2_fusion_int: float = 2.0
    s2_fusion_ssim: float = 2.0
    s2_fusion_mean: float = 0.5
    s2_fusion_contrast: float = 0.5

    s2_reg_weight: float = 0.15
    s2_reg_epe: float = 0.5
    s2_reg_smooth: float = 0.30
    s2_reg_photo: float = 0.0
    s2_reg_edge: float = 0.0
    s2_reg_ll: float = 0.20
    s2_reg_hf: float = 0.10

    reg_scale_weights: Optional[list] = None

    print_freq: int = 20
    val_interval: int = 1
    save_interval: int = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Train WaveLetFusion for fusion and registration")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--stage2-warmup-epochs", type=int, default=None)
    parser.add_argument("--stage1-lr", type=float, default=None)
    parser.add_argument("--stage2-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2-fusion", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    return parser.parse_args()


def build_config_from_args(args):
    cfg = TrainConfig()

    if args.dataset_root is not None:
        cfg.dataset_root = args.dataset_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.vis_dir is not None:
        cfg.vis_dir = args.vis_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.epochs_stage1 is not None:
        cfg.epochs_stage1 = args.epochs_stage1
    if args.epochs_stage2 is not None:
        cfg.epochs_stage2 = args.epochs_stage2
    if args.stage2_warmup_epochs is not None:
        cfg.stage2_warmup_epochs = args.stage2_warmup_epochs
    if args.stage1_lr is not None:
        cfg.stage1_lr = args.stage1_lr
    if args.stage2_lr is not None:
        cfg.stage2_lr = args.stage2_lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.val_ratio is not None:
        cfg.val_ratio = args.val_ratio
    if args.skip_stage1:
        cfg.skip_stage1 = True
    if args.skip_stage2_fusion:
        cfg.skip_stage2_fusion = True
    if args.disable_amp:
        cfg.use_amp = False
    if args.non_deterministic:
        cfg.deterministic = False

    return cfg


def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_config(cfg: TrainConfig, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch_to_device(batch: Dict[str, Any], device):
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True).float()
        else:
            output[key] = value
    return output


def get_optional_mask(batch: Dict[str, Any], candidates, device):
    for key in candidates:
        if key in batch:
            return batch[key].to(device, non_blocking=True).float()
    return None


def to_scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


def autocast_context(device, enabled):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def save_checkpoint(path, epoch, model, optimizer=None, scheduler=None, metric=None, stage=None):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metric": metric,
        "stage": stage,
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_model_only(model, ckpt_path, device, logger):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    logger.info(f"Loaded model from: {ckpt_path}")


def to_vis(img):
    return ((img.detach() + 1.0) / 2.0).clamp(0.0, 1.0)


def make_sample_images_grid(ir, vi, ir_d, vi_d, fused, flow_preds_ir, flow_preds_vi):
    ir_t = ir[0:1]
    vi_t = vi[0:1]
    ir_d_t = ir_d[0:1]
    vi_d_t = vi_d[0:1]
    fused_t = fused[0:1]

    flow_ir = flow_preds_ir[-1][0:1]
    flow_vi = flow_preds_vi[-1][0:1]

    ir_reg = flow_warp(ir_d_t, flow_ir, flow_in_pixel=True)
    vi_reg = flow_warp(vi_d_t, flow_vi, flow_in_pixel=True)

    ir_diff_before = torch.abs(ir_d_t - vi_t)
    ir_diff_after = torch.abs(ir_reg - vi_t)
    vi_diff_before = torch.abs(vi_d_t - ir_t)
    vi_diff_after = torch.abs(vi_reg - ir_t)

    blank = torch.zeros_like(ir_t)

    row1 = torch.cat([to_vis(vi_t), to_vis(ir_d_t), to_vis(ir_reg), to_vis(ir_diff_before), to_vis(ir_diff_after)], dim=3)
    row2 = torch.cat([to_vis(ir_t), to_vis(vi_d_t), to_vis(vi_reg), to_vis(vi_diff_before), to_vis(vi_diff_after)], dim=3)
    row3 = torch.cat([to_vis(ir_t), to_vis(vi_t), to_vis(fused_t), blank, blank], dim=3)

    grid = torch.cat([row1, row2, row3], dim=2)
    return grid.squeeze(0).cpu()


def set_stage_trainable(model: nn.Module, stage: str):
    stage = stage.lower()

    if stage == "stage1":
        trainable_modules = {"fe_ir", "fe_vi", "DM_ir", "DM_vi"}
    elif stage == "stage2_fusion":
        trainable_modules = {"embed", "unembed", "feature_attention", "FN"}
    elif stage == "stage2_joint":
        trainable_modules = None
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if trainable_modules is None:
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        return

    for name, module in model.named_children():
        requires_grad = name in trainable_modules
        for p in module.parameters():
            p.requires_grad = requires_grad
        if requires_grad:
            module.train()
        else:
            module.eval()


def build_stage1_criterion(cfg: TrainConfig, device):
    config = {
        "fusion_grad": 0.0,
        "fusion_int": 0.0,
        "fusion_ssim": 0.0,
        "fusion_mean": 0.0,
        "fusion_contrast": 0.0,
        "reg_epe": cfg.s1_reg_epe,
        "reg_smooth": cfg.s1_reg_smooth,
        "reg_photo": cfg.s1_reg_photo,
        "reg_edge": cfg.s1_reg_edge,
        "reg_ll": cfg.s1_reg_ll,
        "reg_hf": cfg.s1_reg_hf,
        "photo_mode": "disabled",
        "reg_scale_weights": cfg.reg_scale_weights,
        "weight_reg_vs_fusion": 1.0,
    }
    return FusionRegistrationLoss(config=config).to(device)


def build_stage2_criterion(cfg: TrainConfig, device):
    config = {
        "fusion_grad": cfg.s2_fusion_grad,
        "fusion_int": cfg.s2_fusion_int,
        "fusion_ssim": cfg.s2_fusion_ssim,
        "fusion_mean": cfg.s2_fusion_mean,
        "fusion_contrast": cfg.s2_fusion_contrast,
        "reg_epe": cfg.s2_reg_epe,
        "reg_smooth": cfg.s2_reg_smooth,
        "reg_photo": cfg.s2_reg_photo,
        "reg_edge": cfg.s2_reg_edge,
        "reg_ll": cfg.s2_reg_ll,
        "reg_hf": cfg.s2_reg_hf,
        "photo_mode": "disabled",
        "reg_scale_weights": cfg.reg_scale_weights,
        "weight_reg_vs_fusion": cfg.s2_reg_weight,
    }
    return FusionRegistrationLoss(config=config).to(device)


def set_stage1_wavelet_loss_ratio(criterion, epoch: int, cfg: TrainConfig):
    base_ll = cfg.s1_reg_ll
    base_hf = cfg.s1_reg_hf

    if cfg.s1_wavelet_warmup_epochs <= 1:
        ll_ratio = 1.0
        hf_ratio = 1.0
    else:
        progress = min(max((epoch - 1) / float(cfg.s1_wavelet_warmup_epochs - 1), 0.0), 1.0)
        ll_ratio = cfg.s1_ll_start_ratio + (1.0 - cfg.s1_ll_start_ratio) * progress
        hf_ratio = cfg.s1_hf_start_ratio + (1.0 - cfg.s1_hf_start_ratio) * progress

    criterion.reg_crit.w_ll = base_ll * ll_ratio
    criterion.reg_crit.w_hf = base_hf * hf_ratio
    return criterion.reg_crit.w_ll, criterion.reg_crit.w_hf


def compute_stage1_loss(criterion, batch, flow_preds_ir, flow_preds_vi):
    ir = batch["ir"]
    vi = batch["vi"]
    ir_d = batch["ir_d"]
    vi_d = batch["vi_d"]
    ir_flow = batch["ir_flow"]
    vi_flow = batch["vi_flow"]

    gt_valid_ir = get_optional_mask(batch, ["ir_valid", "ir_mask", "ir_flow_mask", "mask_ir", "valid_ir"], ir.device)
    gt_valid_vi = get_optional_mask(batch, ["vi_valid", "vi_mask", "vi_flow_mask", "mask_vi", "valid_vi"], vi.device)

    l_reg, l_epe, l_smooth, l_photo, l_edge, l_ll, l_hf = criterion.reg_crit(
        ir_fixed=ir,
        vi_fixed=vi,
        ir_d=ir_d,
        vi_d=vi_d,
        flow_preds_ir=flow_preds_ir,
        flow_preds_vi=flow_preds_vi,
        gt_flow_ir=ir_flow,
        gt_flow_vi=vi_flow,
        gt_valid_ir=gt_valid_ir,
        gt_valid_vi=gt_valid_vi,
    )

    total_loss = l_reg
    logs = {
        "Total": total_loss,
        "Fusion/Total": total_loss.new_tensor(0.0),
        "Fusion/Grad": total_loss.new_tensor(0.0),
        "Fusion/Int": total_loss.new_tensor(0.0),
        "Fusion/SSIM": total_loss.new_tensor(0.0),
        "Fusion/Mean": total_loss.new_tensor(0.0),
        "Fusion/Contrast": total_loss.new_tensor(0.0),
        "Reg/Total": l_reg,
        "Reg/EPE": l_epe,
        "Reg/Smooth": l_smooth,
        "Reg/Photo": l_photo,
        "Reg/Edge": l_edge,
        "Reg/LL": l_ll,
        "Reg/HF": l_hf,
    }
    return total_loss, logs


def compute_stage2_loss(criterion, batch, fused, flow_preds_ir, flow_preds_vi):
    ir = batch["ir"]
    vi = batch["vi"]
    ir_d = batch["ir_d"]
    vi_d = batch["vi_d"]
    ir_flow = batch["ir_flow"]
    vi_flow = batch["vi_flow"]

    gt_valid_ir = get_optional_mask(batch, ["ir_valid", "ir_mask", "ir_flow_mask", "mask_ir", "valid_ir"], ir.device)
    gt_valid_vi = get_optional_mask(batch, ["vi_valid", "vi_mask", "vi_flow_mask", "mask_vi", "valid_vi"], vi.device)

    total_loss, logs = criterion(
        img_vis_fixed=vi,
        img_ir_fixed=ir,
        img_fused=fused,
        ir_fixed=ir,
        vi_fixed=vi,
        ir_d=ir_d,
        vi_d=vi_d,
        flow_preds_ir=flow_preds_ir,
        flow_preds_vi=flow_preds_vi,
        gt_flow_ir=ir_flow,
        gt_flow_vi=vi_flow,
        gt_valid_ir=gt_valid_ir,
        gt_valid_vi=gt_valid_vi,
    )
    return total_loss, logs


def build_optimizer_stage1(model: nn.Module, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.stage1_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))


def build_optimizer_stage2(model: nn.Module, cfg: TrainConfig):
    fusion_names = {"embed", "unembed", "feature_attention", "FN"}
    shared_names = {"fe_ir", "fe_vi"}
    reg_names = {"DM_ir", "DM_vi"}

    fusion_params = []
    shared_params = []
    reg_params = []
    other_params = []
    seen = set()

    for name, module in model.named_children():
        params = list(module.parameters())
        if not params:
            continue

        if name in fusion_names:
            bucket = fusion_params
        elif name in shared_names:
            bucket = shared_params
        elif name in reg_names:
            bucket = reg_params
        else:
            bucket = other_params

        for p in params:
            if id(p) not in seen:
                bucket.append(p)
                seen.add(id(p))

    for p in model.parameters():
        if id(p) not in seen:
            other_params.append(p)
            seen.add(id(p))

    param_groups = []
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": cfg.stage2_lr, "name": "fusion"})
    if shared_params:
        param_groups.append({"params": shared_params, "lr": cfg.stage2_shared_lr, "name": "shared"})
    if reg_params:
        param_groups.append({"params": reg_params, "lr": cfg.stage2_reg_lr, "name": "reg"})
    if other_params:
        param_groups.append({"params": other_params, "lr": cfg.stage2_lr, "name": "other"})

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))


def build_scheduler(optimizer, cfg: TrainConfig, total_epochs: int):
    if cfg.scheduler_type.lower() == "cosine":
        def lr_lambda(epoch):
            progress = min(epoch / max(total_epochs, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if cfg.scheduler_type.lower() == "step":
        def lr_lambda(epoch):
            return cfg.gamma ** (epoch // cfg.step_size)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler_type: {cfg.scheduler_type}")


def format_lr_groups(optimizer):
    parts = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{i}")
        parts.append(f"{name}={group['lr']:.2e}")
    return ", ".join(parts)


LOG_KEYS = [
    "Total",
    "Fusion/Total",
    "Fusion/Grad",
    "Fusion/Int",
    "Fusion/SSIM",
    "Fusion/Mean",
    "Fusion/Contrast",
    "Reg/Total",
    "Reg/EPE",
    "Reg/Smooth",
    "Reg/Photo",
    "Reg/Edge",
    "Reg/LL",
    "Reg/HF",
]


def init_log_meter():
    return {k: 0.0 for k in LOG_KEYS}


def update_log_meter(meter, logs):
    for k in LOG_KEYS:
        meter[k] += to_scalar(logs[k])


def average_log_meter(meter, n):
    n = max(n, 1)
    return {k: v / n for k, v in meter.items()}


def format_epoch_metrics(metrics):
    return (
        f"Fusion(T={metrics['Fusion/Total']:.4f}, "
        f"G={metrics['Fusion/Grad']:.4f}, I={metrics['Fusion/Int']:.4f}, "
        f"S={metrics['Fusion/SSIM']:.4f}, M={metrics['Fusion/Mean']:.4f}, "
        f"C={metrics['Fusion/Contrast']:.4f}) | "
        f"Reg(T={metrics['Reg/Total']:.4f}, EPE={metrics['Reg/EPE']:.4f}, "
        f"Sm={metrics['Reg/Smooth']:.4f}, Ph={metrics['Reg/Photo']:.4f}, "
        f"Ed={metrics['Reg/Edge']:.4f}, LL={metrics['Reg/LL']:.4f}, "
        f"HF={metrics['Reg/HF']:.4f})"
    )


def log_flow_stats(logger, epoch, pred, gt, prefix="IR"):
    pred = pred.detach()
    gt = gt.detach()
    logger.info(
        f"[Debug E{epoch} {prefix}] "
        f"GT min={gt.min().item():.4f}, max={gt.max().item():.4f}, "
        f"mean_abs={gt.abs().mean().item():.4f} | "
        f"Pred min={pred.min().item():.4f}, max={pred.max().item():.4f}, "
        f"mean_abs={pred.abs().mean().item():.4f}"
    )


def log_fusion_stats(logger, epoch, fused, stage_name):
    fused = fused.detach()
    logger.info(
        f"[Fusion E{epoch} {stage_name}] "
        f"min={fused.min().item():.4f}, max={fused.max().item():.4f}, "
        f"mean={fused.mean().item():.4f}"
    )


def train_one_epoch_stage1(model, loader, criterion, optimizer, scaler, device, epoch, cfg, logger):
    set_stage_trainable(model, "stage1")
    current_ll, current_hf = set_stage1_wavelet_loss_ratio(criterion, epoch, cfg)

    meter = init_log_meter()
    num_steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Stage1]")

    for i, raw_batch in enumerate(pbar):
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.use_amp and device.type == "cuda"
        with autocast_context(device, use_amp):
            fused, flow_preds_ir, flow_preds_vi = model(batch["ir"], batch["vi"], batch["ir_d"], batch["vi_d"])
            total_loss, logs = compute_stage1_loss(criterion, batch, flow_preds_ir, flow_preds_vi)

        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss at Stage1 epoch {epoch}, step {i}")
            continue

        if i == 0:
            logger.info(f"[Stage1 E{epoch}] wavelet: LL={current_ll:.4f}, HF={current_hf:.4f}")
            logger.info(
                f"[ir E{epoch}] min={batch['ir'].min().item():.4f}, "
                f"max={batch['ir'].max().item():.4f}, "
                f"mean={batch['ir'].mean().item():.4f}"
            )
            logger.info(
                f"[vi E{epoch}] min={batch['vi'].min().item():.4f}, "
                f"max={batch['vi'].max().item():.4f}, "
                f"mean={batch['vi'].mean().item():.4f}"
            )
            log_flow_stats(logger, epoch, flow_preds_ir[-1], batch["ir_flow"], prefix="ir")
            log_flow_stats(logger, epoch, flow_preds_vi[-1], batch["vi_flow"], prefix="vi")

        scaler.scale(total_loss).backward()

        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        update_log_meter(meter, logs)
        num_steps += 1

        if num_steps % cfg.print_freq == 0:
            pbar.set_postfix(
                {
                    "Loss": f"{to_scalar(logs['Total']):.4f}",
                    "EPE": f"{to_scalar(logs['Reg/EPE']):.4f}",
                    "Sm": f"{to_scalar(logs['Reg/Smooth']):.4f}",
                    "LL": f"{to_scalar(logs['Reg/LL']):.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    return average_log_meter(meter, num_steps), current_ll, current_hf


def train_one_epoch_stage2(model, loader, criterion, optimizer, scaler, device, epoch, cfg, logger, stage_mode: str):
    if stage_mode == "fusion":
        set_stage_trainable(model, "stage2_fusion")
        stage_name = "Stage2_Fusion"
    elif stage_mode == "joint":
        set_stage_trainable(model, "stage2_joint")
        stage_name = "Stage2_Joint"
    else:
        raise ValueError(f"Unknown stage_mode: {stage_mode}")

    meter = init_log_meter()
    num_steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{stage_name}]")

    for i, raw_batch in enumerate(pbar):
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.use_amp and device.type == "cuda"
        with autocast_context(device, use_amp):
            fused, flow_preds_ir, flow_preds_vi = model(batch["ir"], batch["vi"], batch["ir_d"], batch["vi_d"])
            total_loss, logs = compute_stage2_loss(criterion, batch, fused, flow_preds_ir, flow_preds_vi)

        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss at Stage2 epoch {epoch}, step {i}")
            continue

        if i == 0:
            logger.info(
                f"[ir E{epoch}] min={batch['ir'].min().item():.4f}, "
                f"max={batch['ir'].max().item():.4f}, "
                f"mean={batch['ir'].mean().item():.4f}"
            )
            logger.info(
                f"[vi E{epoch}] min={batch['vi'].min().item():.4f}, "
                f"max={batch['vi'].max().item():.4f}, "
                f"mean={batch['vi'].mean().item():.4f}"
            )
            log_flow_stats(logger, epoch, flow_preds_ir[-1], batch["ir_flow"], prefix="ir")
            log_flow_stats(logger, epoch, flow_preds_vi[-1], batch["vi_flow"], prefix="vi")
            log_fusion_stats(logger, epoch, fused, stage_name)

        scaler.scale(total_loss).backward()

        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        update_log_meter(meter, logs)
        num_steps += 1

        if num_steps % cfg.print_freq == 0:
            pbar.set_postfix(
                {
                    "Loss": f"{to_scalar(logs['Total']):.4f}",
                    "Fus": f"{to_scalar(logs['Fusion/Total']):.4f}",
                    "Grad": f"{to_scalar(logs['Fusion/Grad']):.4f}",
                    "Int": f"{to_scalar(logs['Fusion/Int']):.4f}",
                    "EPE": f"{to_scalar(logs['Reg/EPE']):.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    return average_log_meter(meter, num_steps)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, cfg, logger, stage_name: str):
    model.eval()

    meter = init_log_meter()
    num_steps = 0
    is_stage1 = stage_name.lower() == "stage1"

    for i, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)

        use_amp = cfg.use_amp and device.type == "cuda"
        with autocast_context(device, use_amp):
            fused, flow_preds_ir, flow_preds_vi = model(batch["ir"], batch["vi"], batch["ir_d"], batch["vi_d"])
            if is_stage1:
                total_loss, logs = compute_stage1_loss(criterion, batch, flow_preds_ir, flow_preds_vi)
            else:
                total_loss, logs = compute_stage2_loss(criterion, batch, fused, flow_preds_ir, flow_preds_vi)

        if not torch.isfinite(total_loss):
            logger.warning(f"Skip non-finite val loss at epoch {epoch}, step {i}")
            continue

        update_log_meter(meter, logs)
        num_steps += 1

        if i == 0:
            ir_reg = flow_warp(batch["ir_d"][0:1], flow_preds_ir[-1][0:1], flow_in_pixel=True)
            vi_reg = flow_warp(batch["vi_d"][0:1], flow_preds_vi[-1][0:1], flow_in_pixel=True)

            ir_before = torch.abs(batch["ir_d"][0:1] - batch["vi"][0:1]).mean().item()
            ir_after = torch.abs(ir_reg - batch["vi"][0:1]).mean().item()
            vi_before = torch.abs(batch["vi_d"][0:1] - batch["ir"][0:1]).mean().item()
            vi_after = torch.abs(vi_reg - batch["ir"][0:1]).mean().item()

            logger.info(
                f"[Val E{epoch} {stage_name}] "
                f"ir-branch diff: before={ir_before:.4f}, after={ir_after:.4f} | "
                f"vi-branch diff: before={vi_before:.4f}, after={vi_after:.4f}"
            )

            if cfg.vis_dir:
                imgs_grid = make_sample_images_grid(
                    batch["ir"],
                    batch["vi"],
                    batch["ir_d"],
                    batch["vi_d"],
                    fused,
                    flow_preds_ir,
                    flow_preds_vi,
                )
                save_path = os.path.join(cfg.vis_dir, f"{stage_name.lower()}_epoch_{epoch}_val.png")
                vutils.save_image(imgs_grid, save_path)

    return average_log_meter(meter, num_steps)


def main():
    args = parse_args()
    cfg = build_config_from_args(args)

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)

    logger = setup_logger(cfg.output_dir)
    save_config(cfg, cfg.output_dir)

    if not os.path.isdir(cfg.dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {asdict(cfg)}")
    logger.info(
        f"Resume flags: skip_stage1={cfg.skip_stage1}, "
        f"skip_stage2_fusion={cfg.skip_stage2_fusion}"
    )

    full_dataset = IRVIFusionRegistrationDataset(
        root_dir=cfg.dataset_root,
        ir_dir=cfg.ir_dir,
        vi_dir=cfg.vi_dir,
        ir_d_dir=cfg.ir_d_dir,
        vi_d_dir=cfg.vi_d_dir,
        ir_flow_dir=cfg.ir_flow_dir,
        vi_flow_dir=cfg.vi_flow_dir,
        ir_valid_dir=cfg.ir_valid_dir,
        vi_valid_dir=cfg.vi_valid_dir,
        img_size=(cfg.image_size, cfg.image_size),
    )

    if len(full_dataset) < 2:
        raise ValueError("Dataset must contain at least 2 samples.")

    val_size = max(1, int(len(full_dataset) * cfg.val_ratio))
    val_size = min(val_size, len(full_dataset) - 1)
    train_size = len(full_dataset) - val_size

    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_generator)

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=(cfg.num_workers > 0),
    )

    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    logger.info(
        f"Dataset: {len(full_dataset)} samples | "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )

    model = WaveLetFusion(image_size=cfg.image_size).to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.use_amp and device.type == "cuda"))

    if not cfg.skip_stage1:
        logger.info("=" * 20 + " Stage 1: Registration Pre-training " + "=" * 20)

        criterion_s1 = build_stage1_criterion(cfg, device)
        set_stage_trainable(model, "stage1")
        optimizer_s1 = build_optimizer_stage1(model, cfg)
        scheduler_s1 = build_scheduler(optimizer_s1, cfg, total_epochs=cfg.epochs_stage1)
        best_s1_epe = float("inf")

        for epoch in range(1, cfg.epochs_stage1 + 1):
            train_logs, current_ll, current_hf = train_one_epoch_stage1(
                model,
                train_loader,
                criterion_s1,
                optimizer_s1,
                scaler,
                device,
                epoch,
                cfg,
                logger,
            )

            scheduler_s1.step()

            logger.info(
                f"[S1] Epoch {epoch}/{cfg.epochs_stage1} | "
                f"Train Loss: {train_logs['Total']:.4f} | "
                f"EPE: {train_logs['Reg/EPE']:.4f} | "
                f"Smooth: {train_logs['Reg/Smooth']:.4f} | "
                f"Wavelet(LL={current_ll:.4f}, HF={current_hf:.4f}) | "
                f"LR: {format_lr_groups(optimizer_s1)} | "
                f"{format_epoch_metrics(train_logs)}"
            )

            if epoch % cfg.val_interval == 0:
                val_logs = validate(model, val_loader, criterion_s1, device, epoch, cfg, logger, stage_name="Stage1")

                logger.info(
                    f"[S1] Validation | Loss: {val_logs['Total']:.4f} | "
                    f"EPE: {val_logs['Reg/EPE']:.4f} | "
                    f"Smooth: {val_logs['Reg/Smooth']:.4f} | "
                    f"{format_epoch_metrics(val_logs)}"
                )

                if val_logs["Reg/EPE"] < best_s1_epe:
                    best_s1_epe = val_logs["Reg/EPE"]
                    save_checkpoint(
                        path=os.path.join(cfg.output_dir, "best_reg_model.pth"),
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer_s1,
                        scheduler=scheduler_s1,
                        metric=best_s1_epe,
                        stage="stage1",
                    )
                    logger.info(f"[S1] New best model saved. EPE={best_s1_epe:.4f}")

            if epoch % cfg.save_interval == 0:
                save_checkpoint(
                    path=os.path.join(cfg.output_dir, f"stage1_epoch_{epoch}.pth"),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer_s1,
                    scheduler=scheduler_s1,
                    metric=train_logs["Reg/EPE"],
                    stage="stage1",
                )

    else:
        logger.info("=" * 20 + " Stage 1: SKIPPED " + "=" * 20)
        best_reg_path = os.path.join(cfg.output_dir, "best_reg_model.pth")
        if os.path.exists(best_reg_path):
            load_model_only(model, best_reg_path, device, logger)
        else:
            logger.error(
                f"best_reg_model.pth not found: {best_reg_path}. "
                f"Run Stage 1 first or set skip_stage1=False."
            )
            return

    logger.info("=" * 20 + " Stage 2: Fusion Joint Training " + "=" * 20)

    criterion_s2 = build_stage2_criterion(cfg, device)
    set_stage_trainable(model, "stage2_joint")
    optimizer_s2 = build_optimizer_stage2(model, cfg)

    best_s2_loss = float("inf")
    best_s2_epe = float("inf")
    total_epochs = cfg.epochs_stage1 + cfg.epochs_stage2

    if cfg.skip_stage2_fusion:
        start_local_epoch = cfg.stage2_warmup_epochs + 1
        best_fusion_path = os.path.join(cfg.output_dir, "best_joint_model.pth")

        if os.path.exists(best_fusion_path):
            load_model_only(model, best_fusion_path, device, logger)
            logger.info(
                f"Fusion warmup already completed. Resuming joint stage from local_epoch={start_local_epoch}"
            )
        else:
            logger.warning("best_joint_model.pth not found. Starting Stage2_Fusion from scratch.")
            start_local_epoch = 1

        joint_remaining = cfg.epochs_stage2 - cfg.stage2_warmup_epochs
        scheduler_s2 = build_scheduler(optimizer_s2, cfg, total_epochs=max(joint_remaining, 1))
    else:
        start_local_epoch = 1
        scheduler_s2 = build_scheduler(optimizer_s2, cfg, total_epochs=cfg.epochs_stage2)

        best_reg_path = os.path.join(cfg.output_dir, "best_reg_model.pth")
        if os.path.exists(best_reg_path):
            load_model_only(model, best_reg_path, device, logger)
        else:
            logger.warning("best_reg_model.pth not found. Stage2 will start from current weights.")

    for local_epoch in range(start_local_epoch, cfg.epochs_stage2 + 1):
        global_epoch = cfg.epochs_stage1 + local_epoch

        if local_epoch <= cfg.stage2_warmup_epochs:
            stage_mode = "fusion"
            display_stage = "Stage2_Fusion"
        else:
            stage_mode = "joint"
            display_stage = "Stage2_Joint"

        train_logs = train_one_epoch_stage2(
            model,
            train_loader,
            criterion_s2,
            optimizer_s2,
            scaler,
            device,
            global_epoch,
            cfg,
            logger,
            stage_mode=stage_mode,
        )

        scheduler_s2.step()

        logger.info(
            f"[S2] Epoch {global_epoch}/{total_epochs} ({display_stage}) | "
            f"Train Loss: {train_logs['Total']:.4f} | "
            f"Fusion: {train_logs['Fusion/Total']:.4f} | "
            f"Grad: {train_logs['Fusion/Grad']:.4f} | "
            f"Int: {train_logs['Fusion/Int']:.4f} | "
            f"EPE: {train_logs['Reg/EPE']:.4f} | "
            f"LRs: {format_lr_groups(optimizer_s2)} | "
            f"{format_epoch_metrics(train_logs)}"
        )

        if global_epoch % cfg.val_interval == 0:
            val_logs = validate(model, val_loader, criterion_s2, device, global_epoch, cfg, logger, stage_name="Stage2")

            logger.info(
                f"[S2] Validation | Loss: {val_logs['Total']:.4f} | "
                f"Fusion: {val_logs['Fusion/Total']:.4f} | "
                f"EPE: {val_logs['Reg/EPE']:.4f} | "
                f"{format_epoch_metrics(val_logs)}"
            )

            if val_logs["Total"] < best_s2_loss:
                best_s2_loss = val_logs["Total"]
                save_checkpoint(
                    path=os.path.join(cfg.output_dir, "best_joint_model.pth"),
                    epoch=global_epoch,
                    model=model,
                    optimizer=optimizer_s2,
                    scheduler=scheduler_s2,
                    metric=best_s2_loss,
                    stage="stage2",
                )
                logger.info(f"[S2] New best joint model saved. Loss={best_s2_loss:.4f}")

            if val_logs["Reg/EPE"] < best_s2_epe:
                best_s2_epe = val_logs["Reg/EPE"]
                save_checkpoint(
                    path=os.path.join(cfg.output_dir, "best_joint_epe_model.pth"),
                    epoch=global_epoch,
                    model=model,
                    optimizer=optimizer_s2,
                    scheduler=scheduler_s2,
                    metric=best_s2_epe,
                    stage="stage2",
                )
                logger.info(f"[S2] New best EPE model saved. EPE={best_s2_epe:.4f}")

        if global_epoch % cfg.save_interval == 0:
            save_checkpoint(
                path=os.path.join(cfg.output_dir, f"stage2_epoch_{global_epoch}.pth"),
                epoch=global_epoch,
                model=model,
                optimizer=optimizer_s2,
                scheduler=scheduler_s2,
                metric=train_logs["Total"],
                stage="stage2",
            )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
