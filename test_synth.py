import os
import cv2
import json
import random
import logging
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from network.network import WaveLetFusion, flow_warp


@dataclass
class TestConfig:
    root_ir: str = "./data/ROAD/ir"
    root_vi: str = "./data/ROAD/vi"
    checkpoint_path: str = "./checkpoints/T2.pth"
    output_dir: str = "./outputs/test_res"

    image_size: int = 256
    batch_size: int = 1
    num_workers: int = 4

    aug_times: int = 1
    max_deform: float = 4.0
    sigma_elastic: float = 20.0
    inv_iters: int = 20
    synth_padding_mode: str = "border"

    seed: int = 2024
    max_save: int = 200

    fused_gamma: float = 1.0
    fused_blend_ref: float = 0.3
    fused_p_low: float = 1.0
    fused_p_high: float = 99.0
    use_clahe: bool = True
    clahe_clip: float = 1.5
    clahe_grid: int = 8

    grid_step: int = 16
    grid_line_width: float = 1.2
    grid_alpha: float = 0.85

    grid_sigma: Optional[float] = None
    grid_superscale: int = 2
    grid_post_blur: float = 0.6
    grid_gamma: float = 2.0

    save_vi_deformed: bool = True
    save_original: bool = True


ALIGN_CORNERS = True


def parse_args():
    parser = argparse.ArgumentParser(description="Test WaveLetFusion on synthetic road scene data")
    parser.add_argument("--root-ir", type=str, default=None)
    parser.add_argument("--root-vi", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--aug-times", type=int, default=None)
    parser.add_argument("--max-deform", type=float, default=None)
    parser.add_argument("--sigma-elastic", type=float, default=None)
    parser.add_argument("--inv-iters", type=int, default=None)
    parser.add_argument("--synth-padding-mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-save", type=int, default=None)
    parser.add_argument("--fused-gamma", type=float, default=None)
    parser.add_argument("--fused-blend-ref", type=float, default=None)
    parser.add_argument("--fused-p-low", type=float, default=None)
    parser.add_argument("--fused-p-high", type=float, default=None)
    parser.add_argument("--disable-clahe", action="store_true")
    parser.add_argument("--clahe-clip", type=float, default=None)
    parser.add_argument("--clahe-grid", type=int, default=None)
    parser.add_argument("--grid-step", type=int, default=None)
    parser.add_argument("--grid-line-width", type=float, default=None)
    parser.add_argument("--grid-alpha", type=float, default=None)
    parser.add_argument("--grid-sigma", type=float, default=None)
    parser.add_argument("--grid-superscale", type=int, default=None)
    parser.add_argument("--grid-post-blur", type=float, default=None)
    parser.add_argument("--grid-gamma", type=float, default=None)
    parser.add_argument("--disable-save-vi-deformed", action="store_true")
    parser.add_argument("--disable-save-original", action="store_true")
    return parser.parse_args()


def build_config_from_args(args):
    cfg = TestConfig()

    if args.root_ir is not None:
        cfg.root_ir = args.root_ir
    if args.root_vi is not None:
        cfg.root_vi = args.root_vi
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.aug_times is not None:
        cfg.aug_times = args.aug_times
    if args.max_deform is not None:
        cfg.max_deform = args.max_deform
    if args.sigma_elastic is not None:
        cfg.sigma_elastic = args.sigma_elastic
    if args.inv_iters is not None:
        cfg.inv_iters = args.inv_iters
    if args.synth_padding_mode is not None:
        cfg.synth_padding_mode = args.synth_padding_mode
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_save is not None:
        cfg.max_save = args.max_save
    if args.fused_gamma is not None:
        cfg.fused_gamma = args.fused_gamma
    if args.fused_blend_ref is not None:
        cfg.fused_blend_ref = args.fused_blend_ref
    if args.fused_p_low is not None:
        cfg.fused_p_low = args.fused_p_low
    if args.fused_p_high is not None:
        cfg.fused_p_high = args.fused_p_high
    if args.disable_clahe:
        cfg.use_clahe = False
    if args.clahe_clip is not None:
        cfg.clahe_clip = args.clahe_clip
    if args.clahe_grid is not None:
        cfg.clahe_grid = args.clahe_grid
    if args.grid_step is not None:
        cfg.grid_step = args.grid_step
    if args.grid_line_width is not None:
        cfg.grid_line_width = args.grid_line_width
    if args.grid_alpha is not None:
        cfg.grid_alpha = args.grid_alpha
    if args.grid_sigma is not None:
        cfg.grid_sigma = args.grid_sigma
    if args.grid_superscale is not None:
        cfg.grid_superscale = args.grid_superscale
    if args.grid_post_blur is not None:
        cfg.grid_post_blur = args.grid_post_blur
    if args.grid_gamma is not None:
        cfg.grid_gamma = args.grid_gamma
    if args.disable_save_vi_deformed:
        cfg.save_vi_deformed = False
    if args.disable_save_original:
        cfg.save_original = False

    return cfg


def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("test_synth_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(save_dir, "test.log"), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_config(cfg: TestConfig, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True).float() if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def build_output_dirs(output_dir: str) -> Dict[str, Path]:
    base = Path(output_dir)
    keys = [
        "fused_gray", "fused_color",
        "grids",
        "reg_ir", "reg_vi",
        "flow_ir", "flow_vi",
        "flow_ir_gt", "flow_vi_gt",
        "valid_ir", "valid_vi",
        "ir_deformed",
        "vi_deformed",
        "ir_original",
        "vi_original",
        "grid_overlay",
        "grid_only",
        "diff_abs",
        "diff_grad",
    ]
    dirs: Dict[str, Path] = {}
    for key in keys:
        path = base / key
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs


def save_single_image(img_tensor: torch.Tensor, save_path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(img_tensor.cpu(), str(path))


def save_bgr_image(img_bgr_u8: np.ndarray, save_path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr_u8)


def to_vis(img: torch.Tensor) -> torch.Tensor:
    return ((img.detach() + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_m11_to_u8(img_tensor: torch.Tensor) -> np.ndarray:
    t = img_tensor.detach().cpu()
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.dim() == 4:
        t = t.squeeze(0).squeeze(0)
    arr = t.numpy().astype(np.float32)
    return ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)


def tensor_01_to_u8(img_tensor: torch.Tensor) -> np.ndarray:
    t = img_tensor.detach().cpu()
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    arr = t.numpy().astype(np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def gray_u8_to_tensor01(gray_u8: np.ndarray, device=None) -> torch.Tensor:
    x = torch.from_numpy(gray_u8).float().unsqueeze(0) / 255.0
    return x.to(device) if device is not None else x


def gray_to_bgr(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def normalize_map_to_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


def fused_tensor_to_u8_for_save(
    fused_gray: torch.Tensor,
    ref_gray_u8: Optional[np.ndarray] = None,
    gamma: float = 1.20,
    blend_ref: float = 0.18,
    p_low: float = 1.0,
    p_high: float = 99.0,
    use_clahe: bool = True,
    clahe_clip: float = 1.5,
    clahe_grid: int = 8,
) -> np.ndarray:
    if fused_gray.dim() == 3:
        fused_gray = fused_gray.squeeze(0)

    arr = fused_gray.detach().cpu().numpy().astype(np.float32)
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

    lo, hi = np.percentile(arr, p_low), np.percentile(arr, p_high)
    if hi - lo >= 1e-6:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    arr = np.power(arr, gamma)

    if ref_gray_u8 is not None and blend_ref > 0:
        ref = ref_gray_u8.astype(np.float32) / 255.0
        arr = np.clip((1.0 - blend_ref) * arr + blend_ref * ref, 0.0, 1.0)

    gray_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)

    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(int(clahe_grid), int(clahe_grid))
        )
        gray_u8 = clahe.apply(gray_u8)

    return gray_u8


def colorize_with_visible_ycrcb_from_u8(
    fused_y_u8: np.ndarray, vi_bgr_u8: np.ndarray
) -> np.ndarray:
    vi_ycrcb = cv2.cvtColor(vi_bgr_u8, cv2.COLOR_BGR2YCrCb)
    vi_ycrcb[..., 0] = fused_y_u8
    return cv2.cvtColor(vi_ycrcb, cv2.COLOR_YCrCb2BGR)


def masked_mean(x: torch.Tensor, mask=None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return (x * mask).sum() / (mask.sum() + eps)


def compute_epe(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    valid_mask=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    diff = pred_flow - gt_flow
    epe_map = torch.sqrt(torch.sum(diff * diff, dim=1, keepdim=True) + eps)
    return masked_mean(epe_map, valid_mask)


def pixel_flow_to_norm(
    flow: torch.Tensor, h: int, w: int, align_corners: bool = True
) -> torch.Tensor:
    flow_norm = flow.clone()
    dw = max(w - 1, 1) if align_corners else max(w, 1)
    dh = max(h - 1, 1) if align_corners else max(h, 1)
    flow_norm[:, 0] = flow_norm[:, 0] * 2.0 / dw
    flow_norm[:, 1] = flow_norm[:, 1] * 2.0 / dh
    return flow_norm


def build_base_grid(b, h, w, device, dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)


def warp_by_flow_torch(
    x: torch.Tensor,
    flow_px: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    b, c, h, w = x.shape
    flow_norm = pixel_flow_to_norm(flow_px, h, w, align_corners)
    vgrid = build_base_grid(b, h, w, x.device, x.dtype) + flow_norm.permute(0, 2, 3, 1).contiguous()
    return F.grid_sample(
        x,
        vgrid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def warp_tensor(
    x: torch.Tensor,
    flow_px: torch.Tensor,
    padding_mode: str = "border",
    return_mask: bool = False,
):
    b, c, h, w = x.shape
    flow_norm = pixel_flow_to_norm(flow_px, h, w, ALIGN_CORNERS)
    vgrid = build_base_grid(b, h, w, x.device, x.dtype) + flow_norm.permute(0, 2, 3, 1).contiguous()
    out = F.grid_sample(
        x,
        vgrid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=ALIGN_CORNERS,
    )
    if not return_mask:
        return out
    ones = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    valid = F.grid_sample(
        ones,
        vgrid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=ALIGN_CORNERS,
    )
    return out, (valid > 0.999).float()


def invert_displacement(forward_flow: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    inv = -forward_flow.clone()
    for _ in range(num_iters):
        inv = -warp_tensor(forward_flow, inv, padding_mode="border")
    return inv


def gaussian_kernel1d(kernel_size, sigma, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur_flow(flow: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
    if sigma <= 0:
        return flow
    radius = max(1, int(round(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    _, c, _, _ = flow.shape
    device, dtype = flow.device, flow.dtype
    k1 = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    kx = k1.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)
    ky = k1.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)
    flow = F.conv2d(F.pad(flow, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    flow = F.conv2d(F.pad(flow, (0, 0, radius, radius), mode="reflect"), ky, groups=c)
    return flow


def create_smooth_flow(b, h, w, max_flow=4.0, sigma=20.0, device="cpu") -> torch.Tensor:
    scale = 8
    hs, ws = max(2, h // scale), max(2, w // scale)
    flow = (torch.rand(b, 2, hs, ws, device=device) - 0.5) * 2.0
    flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS)
    flow = gaussian_blur_flow(flow, sigma=sigma)
    flow_max = flow.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return flow / flow_max * max_flow


class SynthRoadSceneDataset(Dataset):
    def __init__(
        self,
        root_ir: str,
        root_vi: str,
        img_size: int = 256,
        aug_times: int = 1,
        max_deform: float = 4.0,
        sigma_elastic: float = 20.0,
        inv_iters: int = 20,
        synth_padding_mode: str = "border",
        seed: int = 2024,
    ):
        super().__init__()
        self.root_ir = root_ir
        self.root_vi = root_vi
        self.img_size = img_size
        self.aug_times = aug_times
        self.max_deform = max_deform
        self.sigma_elastic = sigma_elastic
        self.inv_iters = inv_iters
        self.synth_padding_mode = synth_padding_mode
        self.seed = seed

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.names = sorted(
            [
                f for f in os.listdir(root_ir)
                if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(root_vi, f))
            ]
        )
        if not self.names:
            raise RuntimeError(f"No matched image pairs found:\n  IR={root_ir}\n  VI={root_vi}")

    def __len__(self) -> int:
        return len(self.names) * self.aug_times

    def _read_gray(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _to_tensor_01(img_u8: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img_u8).float().unsqueeze(0) / 255.0

    def __getitem__(self, idx: int) -> dict:
        base_idx = idx // self.aug_times
        aug_idx = idx % self.aug_times
        orig_name = self.names[base_idx]

        img_ir = self._resize(self._read_gray(os.path.join(self.root_ir, orig_name)))
        img_vi = self._resize(self._read_gray(os.path.join(self.root_vi, orig_name)))

        t_ir_01 = self._to_tensor_01(img_ir)
        t_vi_01 = self._to_tensor_01(img_vi)

        torch.manual_seed(self.seed + idx)
        forward_ir = create_smooth_flow(
            1,
            self.img_size,
            self.img_size,
            max_flow=self.max_deform,
            sigma=self.sigma_elastic,
            device="cpu",
        )

        torch.manual_seed(self.seed + idx + 99991)
        forward_vi = create_smooth_flow(
            1,
            self.img_size,
            self.img_size,
            max_flow=self.max_deform,
            sigma=self.sigma_elastic,
            device="cpu",
        )

        ir_d_01 = warp_tensor(t_ir_01.unsqueeze(0), forward_ir, padding_mode=self.synth_padding_mode).squeeze(0)
        vi_d_01 = warp_tensor(t_vi_01.unsqueeze(0), forward_vi, padding_mode=self.synth_padding_mode).squeeze(0)

        gt_ir = invert_displacement(forward_ir, self.inv_iters).squeeze(0)
        gt_vi = invert_displacement(forward_vi, self.inv_iters).squeeze(0)

        _, ir_valid = warp_tensor(
            ir_d_01.unsqueeze(0),
            gt_ir.unsqueeze(0),
            padding_mode="zeros",
            return_mask=True,
        )
        _, vi_valid = warp_tensor(
            vi_d_01.unsqueeze(0),
            gt_vi.unsqueeze(0),
            padding_mode="zeros",
            return_mask=True,
        )

        stem = os.path.splitext(orig_name)[0]
        return {
            "name": f"{stem}_{aug_idx:02d}.png",
            "orig_name": orig_name,
            "ir": t_ir_01 * 2.0 - 1.0,
            "vi": t_vi_01 * 2.0 - 1.0,
            "ir_d": ir_d_01 * 2.0 - 1.0,
            "vi_d": vi_d_01 * 2.0 - 1.0,
            "ir_flow": gt_ir,
            "vi_flow": gt_vi,
            "ir_valid": ir_valid.squeeze(0),
            "vi_valid": vi_valid.squeeze(0),
        }


def flow_to_color(flow_2hw: torch.Tensor) -> torch.Tensor:
    flow = flow_2hw.detach().cpu().numpy()
    fx, fy = flow[0], flow[1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    hsv = np.zeros((*fx.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / mag.max() * 255.0, 0, 255).astype(np.uint8) if mag.max() > 1e-6 else 0
    rgb = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def make_soft_grid_image(
    h: int,
    w: int,
    step: int = 16,
    line_width: float = 1.2,
    sigma: Optional[float] = None,
    superscale: int = 2,
) -> np.ndarray:
    if sigma is None:
        sigma = max(0.4, line_width * 0.9)

    sh, sw = h * superscale, w * superscale
    s_step = step * superscale
    s_sigma = sigma * superscale

    ys = np.arange(sh, dtype=np.float32)
    dist_h = np.mod(ys, s_step)
    dist_h = np.minimum(dist_h, s_step - dist_h)
    dist_h = dist_h.reshape(sh, 1)

    xs = np.arange(sw, dtype=np.float32)
    dist_v = np.mod(xs, s_step)
    dist_v = np.minimum(dist_v, s_step - dist_v)
    dist_v = dist_v.reshape(1, sw)

    dist = np.minimum(
        dist_h * np.ones((sh, sw), dtype=np.float32),
        dist_v * np.ones((sh, sw), dtype=np.float32),
    )

    soft_grid = np.exp(-(dist ** 2) / (2.0 * s_sigma ** 2)).astype(np.float32)

    if superscale > 1:
        soft_grid = cv2.resize(soft_grid, (w, h), interpolation=cv2.INTER_AREA)

    return (soft_grid * 255.0).clip(0, 255).astype(np.uint8)


def warp_soft_grid_image(
    grid_u8: np.ndarray,
    flow_2hw: torch.Tensor,
    padding_mode: str = "zeros",
    post_blur_sigma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    device = flow_2hw.device

    grid_t = torch.from_numpy(grid_u8).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    warped_t = warp_by_flow_torch(
        grid_t,
        flow_2hw.unsqueeze(0),
        padding_mode=padding_mode,
        align_corners=True,
    )

    warped_f32 = warped_t.squeeze().cpu().numpy().astype(np.float32)

    if post_blur_sigma > 0.0:
        ksize = int(2 * round(2.5 * post_blur_sigma) + 1)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        warped_f32 = cv2.GaussianBlur(warped_f32, (ksize, ksize), post_blur_sigma)

    warped_f32 = warped_f32.clip(0.0, 1.0)
    warped_u8 = (warped_f32 * 255.0).astype(np.uint8)
    return warped_u8, warped_f32


def overlay_smooth_grid_on_gray(
    gray_u8: np.ndarray,
    warped_f32: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    grid_alpha: float = 0.85,
    gamma_grid: float = 1.2,
) -> np.ndarray:
    base_bgr = gray_to_bgr(gray_u8).astype(np.float32)
    weight = np.power(warped_f32, gamma_grid)
    weight = (weight * grid_alpha)[..., np.newaxis]
    color_layer = np.full_like(base_bgr, color, dtype=np.float32)
    blended = weight * color_layer + (1.0 - weight) * base_bgr
    return blended.clip(0, 255).astype(np.uint8)


def make_smooth_deformation_grid_overlay(
    img_tensor_m11: torch.Tensor,
    flow_2hw: torch.Tensor,
    step: int = 16,
    line_width: float = 1.2,
    sigma: Optional[float] = None,
    superscale: int = 2,
    post_blur_sigma: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 0),
    grid_alpha: float = 0.85,
    gamma_grid: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_u8 = tensor_m11_to_u8(img_tensor_m11)
    h, w = gray_u8.shape

    soft_grid = make_soft_grid_image(
        h,
        w,
        step=step,
        line_width=line_width,
        sigma=sigma,
        superscale=superscale,
    )

    warped_u8, warped_f32 = warp_soft_grid_image(
        soft_grid,
        flow_2hw,
        padding_mode="zeros",
        post_blur_sigma=post_blur_sigma,
    )

    overlay = overlay_smooth_grid_on_gray(
        gray_u8,
        warped_f32,
        color=color,
        grid_alpha=grid_alpha,
        gamma_grid=gamma_grid,
    )

    return gray_to_bgr(gray_u8), overlay, warped_u8


def save_grid_visualization(
    dirs: Dict[str, Path],
    stem: str,
    img_tensor_m11: torch.Tensor,
    flow_2hw: torch.Tensor,
    tag: str,
    cfg: TestConfig,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    _, overlay, warped_grid = make_smooth_deformation_grid_overlay(
        img_tensor_m11=img_tensor_m11,
        flow_2hw=flow_2hw,
        step=cfg.grid_step,
        line_width=float(getattr(cfg, "grid_line_width", 1.2)),
        sigma=getattr(cfg, "grid_sigma", None),
        superscale=int(getattr(cfg, "grid_superscale", 2)),
        post_blur_sigma=float(getattr(cfg, "grid_post_blur", 0.6)),
        color=color,
        grid_alpha=cfg.grid_alpha,
        gamma_grid=float(getattr(cfg, "grid_gamma", 1.2)),
    )
    cv2.imwrite(str(dirs["grid_overlay"] / f"{stem}_{tag}.png"), overlay)
    cv2.imwrite(str(dirs["grid_only"] / f"{stem}_{tag}.png"), warped_grid)


def sobel_grad_mag(gray_u8: np.ndarray) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def make_abs_diff_map(img1, img2) -> np.ndarray:
    d = np.abs(tensor_m11_to_u8(img1).astype(np.float32) - tensor_m11_to_u8(img2).astype(np.float32))
    return normalize_map_to_u8(d)


def make_gradient_diff_map(img1, img2) -> np.ndarray:
    d = np.abs(sobel_grad_mag(tensor_m11_to_u8(img1)) - sobel_grad_mag(tensor_m11_to_u8(img2)))
    return normalize_map_to_u8(d)


def colorize_heatmap_on_gray(gray_u8, heat_u8, alpha=0.45, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    return cv2.addWeighted(
        gray_to_bgr(gray_u8),
        1.0 - alpha,
        cv2.applyColorMap(heat_u8, cmap),
        alpha,
        0.0,
    )


def save_diff_visualizations(dirs: Dict[str, Path], stem: str, moving, target, tag: str) -> None:
    abs_diff = make_abs_diff_map(moving, target)
    grad_diff = make_gradient_diff_map(moving, target)
    base_u8 = tensor_m11_to_u8(moving)

    cv2.imwrite(
        str(dirs["diff_abs"] / f"{stem}_{tag}_overlay.png"),
        colorize_heatmap_on_gray(base_u8, abs_diff, alpha=0.45, cmap=cv2.COLORMAP_JET),
    )
    cv2.imwrite(str(dirs["diff_abs"] / f"{stem}_{tag}_map.png"), abs_diff)

    cv2.imwrite(
        str(dirs["diff_grad"] / f"{stem}_{tag}_overlay.png"),
        colorize_heatmap_on_gray(base_u8, grad_diff, alpha=0.45, cmap=cv2.COLORMAP_TURBO),
    )
    cv2.imwrite(str(dirs["diff_grad"] / f"{stem}_{tag}_map.png"), grad_diff)


def make_sample_images_grid(ir, vi, ir_d, vi_d, fused_01, flow_preds_ir, flow_preds_vi) -> torch.Tensor:
    ir_t, vi_t = ir[0:1], vi[0:1]
    ird_t, vid_t = ir_d[0:1], vi_d[0:1]
    fused_t = fused_01[0:1].to(ir_t.device)

    flow_ir = flow_preds_ir[-1][0:1]
    flow_vi = flow_preds_vi[-1][0:1]

    ir_reg = flow_warp(ird_t, flow_ir, flow_in_pixel=True)
    vi_reg = flow_warp(vid_t, flow_vi, flow_in_pixel=True)

    blank = torch.zeros_like(fused_t)

    row1 = torch.cat(
        [
            to_vis(vi_t),
            to_vis(ird_t),
            to_vis(ir_reg),
            torch.abs(to_vis(ird_t) - to_vis(vi_t)),
            torch.abs(to_vis(ir_reg) - to_vis(vi_t)),
        ],
        dim=3,
    )
    row2 = torch.cat(
        [
            to_vis(ir_t),
            to_vis(vid_t),
            to_vis(vi_reg),
            torch.abs(to_vis(vid_t) - to_vis(ir_t)),
            torch.abs(to_vis(vi_reg) - to_vis(ir_t)),
        ],
        dim=3,
    )
    row3 = torch.cat([to_vis(ir_t), to_vis(vi_t), fused_t, blank, blank], dim=3)

    return torch.cat([row1, row2, row3], dim=2).squeeze(0).cpu()


def save_sample_outputs(
    b: int,
    stem: str,
    orig_name: str,
    batch: dict,
    pred_ir: torch.Tensor,
    pred_vi: torch.Tensor,
    fused: torch.Tensor,
    flow_preds_ir: List[torch.Tensor],
    flow_preds_vi: List[torch.Tensor],
    dirs: Dict[str, Path],
    cfg: TestConfig,
) -> None:
    device = batch["ir"].device

    ir_d_u8 = tensor_m11_to_u8(batch["ir_d"][b])
    cv2.imwrite(str(dirs["ir_deformed"] / f"{stem}_ir_deformed.png"), gray_to_bgr(ir_d_u8))

    if cfg.save_vi_deformed:
        vi_d_u8 = tensor_m11_to_u8(batch["vi_d"][b])
        cv2.imwrite(str(dirs["vi_deformed"] / f"{stem}_vi_deformed.png"), gray_to_bgr(vi_d_u8))

    if cfg.save_original:
        cv2.imwrite(str(dirs["ir_original"] / f"{stem}_ir.png"), gray_to_bgr(tensor_m11_to_u8(batch["ir"][b])))
        cv2.imwrite(str(dirs["vi_original"] / f"{stem}_vi.png"), gray_to_bgr(tensor_m11_to_u8(batch["vi"][b])))

    vi_color_path = Path(cfg.root_vi) / orig_name
    vi_color_bgr = cv2.imread(str(vi_color_path), cv2.IMREAD_COLOR)
    vi_y: Optional[np.ndarray] = None
    if vi_color_bgr is not None:
        vi_color_bgr = cv2.resize(
            vi_color_bgr,
            (cfg.image_size, cfg.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        vi_y = cv2.cvtColor(vi_color_bgr, cv2.COLOR_BGR2YCrCb)[..., 0]

    fused_gray_u8 = fused_tensor_to_u8_for_save(
        fused[b],
        ref_gray_u8=vi_y,
        gamma=cfg.fused_gamma,
        blend_ref=cfg.fused_blend_ref,
        p_low=cfg.fused_p_low,
        p_high=cfg.fused_p_high,
        use_clahe=cfg.use_clahe,
        clahe_clip=cfg.clahe_clip,
        clahe_grid=cfg.clahe_grid,
    )
    fused_gray_img = gray_u8_to_tensor01(fused_gray_u8, device=device)
    save_single_image(fused_gray_img, dirs["fused_gray"] / f"{stem}_fused_gray.png")

    if vi_color_bgr is not None:
        fused_color_bgr = colorize_with_visible_ycrcb_from_u8(fused_gray_u8, vi_color_bgr)
        cv2.imwrite(str(dirs["fused_color"] / f"{stem}_fused_color.png"), fused_color_bgr)

    ir_reg = flow_warp(batch["ir_d"][b:b + 1], pred_ir[b:b + 1], flow_in_pixel=True)
    vi_reg = flow_warp(batch["vi_d"][b:b + 1], pred_vi[b:b + 1], flow_in_pixel=True)
    save_single_image(to_vis(ir_reg[0]), dirs["reg_ir"] / f"{stem}_ir_reg.png")
    save_single_image(to_vis(vi_reg[0]), dirs["reg_vi"] / f"{stem}_vi_reg.png")

    gt_ir, gt_vi = batch["ir_flow"], batch["vi_flow"]
    save_single_image(flow_to_color(pred_ir[b]), dirs["flow_ir"] / f"{stem}_flow_ir.png")
    save_single_image(flow_to_color(pred_vi[b]), dirs["flow_vi"] / f"{stem}_flow_vi.png")
    save_single_image(flow_to_color(gt_ir[b]), dirs["flow_ir_gt"] / f"{stem}_flow_ir_gt.png")
    save_single_image(flow_to_color(gt_vi[b]), dirs["flow_vi_gt"] / f"{stem}_flow_vi_gt.png")

    save_single_image(batch["ir_valid"][b].cpu().clamp(0, 1), dirs["valid_ir"] / f"{stem}_valid_ir.png")
    save_single_image(batch["vi_valid"][b].cpu().clamp(0, 1), dirs["valid_vi"] / f"{stem}_valid_vi.png")

    grid = make_sample_images_grid(
        batch["ir"][b:b + 1],
        batch["vi"][b:b + 1],
        batch["ir_d"][b:b + 1],
        batch["vi_d"][b:b + 1],
        fused_gray_img.unsqueeze(0),
        [x[b:b + 1] for x in flow_preds_ir],
        [x[b:b + 1] for x in flow_preds_vi],
    )
    save_single_image(grid, dirs["grids"] / f"{stem}_grid.png")

    save_grid_visualization(dirs, stem, batch["ir_d"][b], pred_ir[b], "pred_grid_ir", cfg, color=(0, 255, 0))
    save_grid_visualization(dirs, stem, batch["vi_d"][b], pred_vi[b], "pred_grid_vi", cfg, color=(0, 255, 0))
    save_grid_visualization(dirs, stem, batch["ir_d"][b], gt_ir[b], "gt_grid_ir", cfg, color=(0, 0, 255))
    save_grid_visualization(dirs, stem, batch["vi_d"][b], gt_vi[b], "gt_grid_vi", cfg, color=(0, 0, 255))

    save_diff_visualizations(dirs, stem, batch["ir_d"][b], batch["vi"][b], "before_ir")
    save_diff_visualizations(dirs, stem, batch["vi_d"][b], batch["ir"][b], "before_vi")
    save_diff_visualizations(dirs, stem, ir_reg[0], batch["vi"][b], "after_ir")
    save_diff_visualizations(dirs, stem, vi_reg[0], batch["ir"][b], "after_vi")


def load_checkpoint(model, ckpt_path: str, device, logger) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
        for key in ("epoch", "stage", "metric"):
            if key in ckpt:
                logger.info(f"Checkpoint {key}: {ckpt[key]}")
    else:
        model.load_state_dict(ckpt, strict=True)
    logger.info(f"Loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def evaluate(model, loader, device, cfg: TestConfig, logger) -> dict:
    model.eval()

    dirs = build_output_dirs(cfg.output_dir)
    max_save = cfg.max_save if (cfg.max_save and cfg.max_save > 0) else int(1e12)

    total = 0
    sums = dict(
        epe_ir=0.0,
        epe_vi=0.0,
        ir_before=0.0,
        ir_after=0.0,
        vi_before=0.0,
        vi_after=0.0,
        fused_min=0.0,
        fused_max=0.0,
        fused_mean=0.0,
    )

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Evaluating", dynamic_ncols=True)

    for idx, raw_batch in pbar:
        batch = move_batch_to_device(raw_batch, device)
        names = [batch["name"]] if isinstance(batch["name"], str) else list(batch["name"])
        orig_names = [batch["orig_name"]] if isinstance(batch["orig_name"], str) else list(batch["orig_name"])

        fused, flow_preds_ir, flow_preds_vi = model(batch["ir"], batch["vi"], batch["ir_d"], batch["vi_d"])
        pred_ir = flow_preds_ir[-1]
        pred_vi = flow_preds_vi[-1]
        gt_ir, gt_vi = batch["ir_flow"], batch["vi_flow"]

        epe_ir = compute_epe(pred_ir, gt_ir, batch["ir_valid"])
        epe_vi = compute_epe(pred_vi, gt_vi, batch["vi_valid"])

        ir_reg = flow_warp(batch["ir_d"], pred_ir, flow_in_pixel=True)
        vi_reg = flow_warp(batch["vi_d"], pred_vi, flow_in_pixel=True)

        bsz = batch["ir"].shape[0]
        sums["epe_ir"] += epe_ir.item() * bsz
        sums["epe_vi"] += epe_vi.item() * bsz
        sums["ir_before"] += torch.abs(batch["ir_d"] - batch["vi"]).mean().item() * bsz
        sums["ir_after"] += torch.abs(ir_reg - batch["vi"]).mean().item() * bsz
        sums["vi_before"] += torch.abs(batch["vi_d"] - batch["ir"]).mean().item() * bsz
        sums["vi_after"] += torch.abs(vi_reg - batch["ir"]).mean().item() * bsz
        sums["fused_min"] += fused.min().item() * bsz
        sums["fused_max"] += fused.max().item() * bsz
        sums["fused_mean"] += fused.mean().item() * bsz
        total += bsz

        if idx < max_save:
            for b in range(bsz):
                stem = os.path.splitext(names[b])[0]
                save_sample_outputs(
                    b=b,
                    stem=stem,
                    orig_name=orig_names[b],
                    batch=batch,
                    pred_ir=pred_ir,
                    pred_vi=pred_vi,
                    fused=fused,
                    flow_preds_ir=flow_preds_ir,
                    flow_preds_vi=flow_preds_vi,
                    dirs=dirs,
                    cfg=cfg,
                )

        pbar.set_postfix(
            {
                "EPE_IR": f"{sums['epe_ir'] / total:.4f}",
                "EPE_VI": f"{sums['epe_vi'] / total:.4f}",
            }
        )

    return {
        "num_samples": total,
        "EPE_IR": sums["epe_ir"] / total,
        "EPE_VI": sums["epe_vi"] / total,
        "EPE_Mean": (sums["epe_ir"] + sums["epe_vi"]) / (2 * total),
        "IR_Before": sums["ir_before"] / total,
        "IR_After": sums["ir_after"] / total,
        "VI_Before": sums["vi_before"] / total,
        "VI_After": sums["vi_after"] / total,
        "Fused_Min": sums["fused_min"] / total,
        "Fused_Max": sums["fused_max"] / total,
        "Fused_Mean": sums["fused_mean"] / total,
    }


def main():
    args = parse_args()
    cfg = build_config_from_args(args)
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logger(cfg.output_dir)
    save_config(cfg, cfg.output_dir)

    if not os.path.isdir(cfg.root_ir):
        raise FileNotFoundError(f"IR directory not found: {cfg.root_ir}")
    if not os.path.isdir(cfg.root_vi):
        raise FileNotFoundError(f"VI directory not found: {cfg.root_vi}")
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    logger.info(f"Config: {asdict(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = SynthRoadSceneDataset(
        root_ir=cfg.root_ir,
        root_vi=cfg.root_vi,
        img_size=cfg.image_size,
        aug_times=cfg.aug_times,
        max_deform=cfg.max_deform,
        sigma_elastic=cfg.sigma_elastic,
        inv_iters=cfg.inv_iters,
        synth_padding_mode=cfg.synth_padding_mode,
        seed=cfg.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    model = WaveLetFusion(image_size=cfg.image_size).to(device)
    load_checkpoint(model, cfg.checkpoint_path, device, logger)

    results = evaluate(model=model, loader=loader, device=device, cfg=cfg, logger=logger)

    logger.info("=" * 60)
    logger.info("Test Finished")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
