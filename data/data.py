# -*- coding: utf-8 -*-
import os
import cv2
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ================= 配置区域 =================
ROOT_IR = r"/tmp/pycharm_project_709/medical/SPECT-MRI/SPECT"
ROOT_VI = r"/tmp/pycharm_project_709/medical/SPECT-MRI/MRI"

OUTPUT_ROOT = r"/tmp/pycharm_project_709/data/MRI_SPECT_Train"

IMG_SIZE = 256
AUG_TIMES = 8
MAX_DEFORM = 4.0
SIGMA_ELASTIC = 20.0
SEED = 2024

ALIGN_CORNERS = True
SYNTH_PADDING_MODE = "border"   # 生成 moving 图时建议用 border
INV_ITERS = 20                  # 逆场迭代次数
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dirs(root, subdirs):
    for d in subdirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def gaussian_kernel1d(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_flow(flow, sigma=10.0):
    """
    flow: [B, 2, H, W]
    """
    if sigma <= 0:
        return flow

    radius = max(1, int(round(3.0 * sigma)))
    kernel_size = 2 * radius + 1

    b, c, h, w = flow.shape
    device, dtype = flow.device, flow.dtype

    k1 = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    kx = k1.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)
    ky = k1.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)

    flow = F.pad(flow, (radius, radius, 0, 0), mode='reflect')
    flow = F.conv2d(flow, kx, groups=c)

    flow = F.pad(flow, (0, 0, radius, radius), mode='reflect')
    flow = F.conv2d(flow, ky, groups=c)

    return flow


def create_smooth_flow(b, h, w, max_flow=4.0, sigma=20.0, device="cpu"):
    """
    生成平滑随机弹性位移场
    返回: [B, 2, H, W]，单位: pixel
    """
    scale = 8
    hs = max(2, h // scale)
    ws = max(2, w // scale)

    flow = (torch.rand(b, 2, hs, ws, device=device) - 0.5) * 2.0
    flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
    flow = gaussian_blur_flow(flow, sigma=sigma)

    flow_max = flow.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    flow = flow / flow_max * max_flow
    return flow


def pixel_flow_to_norm(flow, h, w, align_corners=True):
    """
    flow: [B, 2, H, W] pixel -> normalized
    """
    flow_norm = flow.clone()
    if align_corners:
        flow_norm[:, 0] = flow_norm[:, 0] * 2.0 / max(w - 1, 1)
        flow_norm[:, 1] = flow_norm[:, 1] * 2.0 / max(h - 1, 1)
    else:
        flow_norm[:, 0] = flow_norm[:, 0] * 2.0 / max(w, 1)
        flow_norm[:, 1] = flow_norm[:, 1] * 2.0 / max(h, 1)
    return flow_norm


def base_grid_norm(b, h, w, device, dtype):
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    return grid


def warp_tensor(x, flow_px, padding_mode="border", return_mask=False):
    """
    x:       [B, C, H, W]
    flow_px: [B, 2, H, W]，定义在输出网格上，单位 pixel
    """
    b, c, h, w = x.shape
    flow_norm = pixel_flow_to_norm(flow_px, h, w, align_corners=ALIGN_CORNERS)
    grid = base_grid_norm(b, h, w, x.device, x.dtype)
    vgrid = grid + flow_norm.permute(0, 2, 3, 1).contiguous()

    out = F.grid_sample(
        x, vgrid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=ALIGN_CORNERS
    )

    if not return_mask:
        return out

    ones = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    valid = F.grid_sample(
        ones, vgrid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=ALIGN_CORNERS
    )
    valid = (valid > 0.999).float()
    return out, valid


def invert_displacement(forward_flow, num_iters=20):
    """
    迭代求逆位移场。
    如果 moving = warp(fixed, forward_flow)
    那么把 moving warp 回 fixed 的 GT 应该是 inverse_flow，而不是简单的 -forward_flow。

    forward_flow / inverse_flow 形状: [B, 2, H, W]
    """
    inv = -forward_flow.clone()
    for _ in range(num_iters):
        inv = -warp_tensor(forward_flow, inv, padding_mode='border')
    return inv


def random_crop_pair(img_ir, img_vi, crop_size=256):
    h, w = img_ir.shape[:2]

    if h < crop_size or w < crop_size:
        scale = max(crop_size / h, crop_size / w) + 0.05
        new_h, new_w = int(h * scale), int(w * scale)
        img_ir = cv2.resize(img_ir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_vi = cv2.resize(img_vi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w

    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)

    return (
        img_ir[y:y + crop_size, x:x + crop_size],
        img_vi[y:y + crop_size, x:x + crop_size]
    )


def save_img_u8(folder, name, tensor_1chw):
    arr = (tensor_1chw.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(folder, name), arr)


def save_mask_u8(folder, name, mask_1chw):
    arr = (mask_1chw.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(folder, name), arr)


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(ROOT_IR) or not os.path.exists(ROOT_VI):
        print("Input folder does not exist.")
        print("ROOT_IR:", ROOT_IR)
        print("ROOT_VI:", ROOT_VI)
        return

    subdirs = [
        "ir", "vi", "ir_d", "vi_d",
        "ir_flows", "vi_flows",
        "ir_valid", "vi_valid"
    ]
    make_dirs(OUTPUT_ROOT, subdirs)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    ir_files = sorted([f for f in os.listdir(ROOT_IR) if f.lower().endswith(valid_exts)])

    if len(ir_files) == 0:
        print(f"No images found in {ROOT_IR}")
        return

    count = 0
    skipped = 0

    pbar = tqdm(ir_files, desc="Generating dataset")
    with torch.no_grad():
        for fname in pbar:
            path_ir = os.path.join(ROOT_IR, fname)
            path_vi = os.path.join(ROOT_VI, fname)

            if not os.path.exists(path_vi):
                skipped += 1
                continue

            img_ir = cv2.imread(path_ir, cv2.IMREAD_GRAYSCALE)
            img_vi = cv2.imread(path_vi, cv2.IMREAD_GRAYSCALE)

            if img_ir is None or img_vi is None:
                skipped += 1
                continue

            for i in range(AUG_TIMES):
                try:
                    patch_ir, patch_vi = random_crop_pair(img_ir, img_vi, IMG_SIZE)

                    t_ir = torch.from_numpy(patch_ir).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    t_vi = torch.from_numpy(patch_vi).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                    # 生成 forward deformation（用于合成 moving）
                    forward_ir = create_smooth_flow(
                        1, IMG_SIZE, IMG_SIZE, max_flow=MAX_DEFORM, sigma=SIGMA_ELASTIC, device=device
                    )
                    forward_vi = create_smooth_flow(
                        1, IMG_SIZE, IMG_SIZE, max_flow=MAX_DEFORM, sigma=SIGMA_ELASTIC, device=device
                    )

                    # 合成 moving 图
                    ir_d = warp_tensor(t_ir, forward_ir, padding_mode=SYNTH_PADDING_MODE)
                    vi_d = warp_tensor(t_vi, forward_vi, padding_mode=SYNTH_PADDING_MODE)

                    # 迭代求 moving -> fixed 的逆场（训练 GT）
                    gt_ir = invert_displacement(forward_ir, num_iters=INV_ITERS)
                    gt_vi = invert_displacement(forward_vi, num_iters=INV_ITERS)

                    # 有效 mask：表示 warp(moving, gt) 时哪些 fixed 像素是真实可采样的
                    _, ir_valid = warp_tensor(ir_d, gt_ir, padding_mode='zeros', return_mask=True)
                    _, vi_valid = warp_tensor(vi_d, gt_vi, padding_mode='zeros', return_mask=True)

                    base_name = os.path.splitext(fname)[0]
                    save_name = f"{base_name}_{i:02d}.png"
                    npy_name = f"{base_name}_{i:02d}.npy"

                    save_img_u8(os.path.join(OUTPUT_ROOT, "ir"), save_name, t_ir)
                    save_img_u8(os.path.join(OUTPUT_ROOT, "vi"), save_name, t_vi)
                    save_img_u8(os.path.join(OUTPUT_ROOT, "ir_d"), save_name, ir_d)
                    save_img_u8(os.path.join(OUTPUT_ROOT, "vi_d"), save_name, vi_d)

                    np.save(os.path.join(OUTPUT_ROOT, "ir_flows", npy_name), gt_ir.squeeze(0).cpu().numpy().astype(np.float32))
                    np.save(os.path.join(OUTPUT_ROOT, "vi_flows", npy_name), gt_vi.squeeze(0).cpu().numpy().astype(np.float32))

                    save_mask_u8(os.path.join(OUTPUT_ROOT, "ir_valid"), save_name, ir_valid)
                    save_mask_u8(os.path.join(OUTPUT_ROOT, "vi_valid"), save_name, vi_valid)

                    count += 1

                except Exception as e:
                    print(f"[Error] {fname} aug#{i}: {e}")
                    continue

    print("\n" + "=" * 40)
    print("Done")
    print(f"Original pairs: {len(ir_files)}")
    print(f"Generated pairs: {count}")
    print(f"Skipped pairs:   {skipped}")
    print(f"Output root:     {os.path.abspath(OUTPUT_ROOT)}")
    print("=" * 40)


if __name__ == "__main__":
    main()
