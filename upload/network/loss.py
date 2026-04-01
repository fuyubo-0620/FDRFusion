import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

ALIGN_CORNERS = True
PADDING_MODE = "reflection"

_WINDOW_CACHE = {}


def gaussian(window_size, sigma):
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def create_window(window_size, channel, device=None, dtype=torch.float32):
    key = (window_size, channel, str(device), str(dtype))
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]

    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()

    if device is not None:
        window = window.to(device=device, dtype=dtype)

    _WINDOW_CACHE[key] = window
    return window


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def _ensure_flow_2chw(flow):
    if flow is None:
        return None

    if flow.dim() != 4:
        raise ValueError(f"Flow must be 4D [B,2,H,W] or [B,H,W,2], got {flow.shape}")

    if flow.shape[1] == 2:
        return flow

    if flow.shape[-1] == 2:
        return flow.permute(0, 3, 1, 2).contiguous()

    raise ValueError(f"Unknown flow shape: {flow.shape}. Expected channel dim 2 at dim=1 or dim=-1.")


def _ensure_flow_list(flow_preds):
    if isinstance(flow_preds, torch.Tensor):
        return [_ensure_flow_2chw(flow_preds)]

    if isinstance(flow_preds, (list, tuple)):
        if len(flow_preds) == 0:
            raise ValueError("flow_preds is empty.")
        return [_ensure_flow_2chw(f) for f in flow_preds]

    raise TypeError(f"flow_preds must be Tensor / list / tuple, but got {type(flow_preds)}")


def _ensure_mask(mask, ref_tensor):
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"mask must be [B,1,H,W] or [B,H,W], got {mask.shape}")
    if mask.shape[1] != 1:
        raise ValueError(f"mask channel must be 1, got {mask.shape}")
    return mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def masked_mean(x, mask=None, eps=1e-6):
    if mask is None:
        return x.mean()
    mask = mask.to(device=x.device, dtype=x.dtype)
    return (x * mask).sum() / (mask.sum() + eps)


def charbonnier_map(x, y, eps=1e-3):
    diff = x - y
    return torch.sqrt(diff * diff + eps * eps)


def charbonnier_loss(x, y, eps=1e-3, mask=None):
    loss_map = charbonnier_map(x, y, eps=eps)
    return masked_mean(loss_map, mask=mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return ssim_map.mean() if size_average else ssim_map


def image_dx(x):
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def image_dy(x):
    return x[:, :, 1:, :] - x[:, :, :-1, :]


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        kernely = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)

    def forward(self, x):
        b, c, h, w = x.shape
        weightx = self.weightx.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        weighty = self.weighty.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, weightx, padding=1, groups=c)
        grad_y = F.conv2d(x, weighty, padding=1, groups=c)

        return torch.abs(grad_x) + torch.abs(grad_y)


class HaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if (x.shape[-2] % 2 != 0) or (x.shape[-1] % 2 != 0):
            raise ValueError(f"H and W must be even for HaarDWT2D, but got {x.shape}")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        ll = torch.nan_to_num(ll, nan=0.0, posinf=0.0, neginf=0.0)
        lh = torch.nan_to_num(lh, nan=0.0, posinf=0.0, neginf=0.0)
        hl = torch.nan_to_num(hl, nan=0.0, posinf=0.0, neginf=0.0)
        hh = torch.nan_to_num(hh, nan=0.0, posinf=0.0, neginf=0.0)
        return ll, lh, hl, hh


def downsample_mask(mask, times=1):
    if mask is None:
        return None

    out = mask
    for _ in range(times):
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
    return out.clamp(0.0, 1.0)


class Fusion_loss(nn.Module):
    def __init__(self, w_grad=10.0, w_int=1.0, w_ssim=5.0, w_mean=1.0, w_contrast=1.0):
        super().__init__()
        self.w_grad = w_grad
        self.w_int = w_int
        self.w_ssim = w_ssim
        self.w_mean = w_mean
        self.w_contrast = w_contrast

        self.sobel = Sobelxy()
        self.window_size = 11

    def _global_mean(self, x):
        return x.mean(dim=(2, 3), keepdim=True)

    def _global_std(self, x):
        return x.std(dim=(2, 3), keepdim=True, unbiased=False)

    def forward(self, img_vis, img_ir, img_fused):
        img_vis = img_vis.clamp(-1.0, 1.0)
        img_ir = img_ir.clamp(-1.0, 1.0)
        img_fused = img_fused.clamp(-1.0, 1.0)

        grad_vis = self.sobel(img_vis)
        grad_ir = self.sobel(img_ir)
        grad_fused = self.sobel(img_fused)

        grad_joint = torch.max(grad_vis, grad_ir)
        loss_grad = charbonnier_loss(grad_fused, grad_joint)

        int_target = 0.5 * (img_vis + img_ir)
        loss_int = charbonnier_loss(img_fused, int_target)

        img_vis_01 = to_01(img_vis)
        img_ir_01 = to_01(img_ir)
        img_fused_01 = to_01(img_fused)

        window = create_window(
            self.window_size,
            img_vis.shape[1],
            device=img_vis.device,
            dtype=img_vis.dtype,
        )

        ssim_vis = _ssim(img_fused_01, img_vis_01, window, self.window_size, img_vis.shape[1])
        ssim_ir = _ssim(img_fused_01, img_ir_01, window, self.window_size, img_ir.shape[1])
        loss_ssim = 1.0 - (ssim_vis + ssim_ir) / 2.0

        mean_target = 0.5 * (self._global_mean(img_vis) + self._global_mean(img_ir))
        mean_fused = self._global_mean(img_fused)
        loss_mean = torch.abs(mean_fused - mean_target).mean()

        std_target = 0.5 * (self._global_std(img_vis) + self._global_std(img_ir))
        std_fused = self._global_std(img_fused)
        loss_contrast = torch.abs(std_fused - std_target).mean()

        total = (
            self.w_grad * loss_grad +
            self.w_int * loss_int +
            self.w_ssim * loss_ssim +
            self.w_mean * loss_mean +
            self.w_contrast * loss_contrast
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)

        return total, loss_grad, loss_int, loss_ssim, loss_mean, loss_contrast


def warp(x, flow, return_mask=False):
    flow = _ensure_flow_2chw(flow).type_as(x)

    b, c, h, w = x.size()

    xx = torch.arange(0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    yy = torch.arange(0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    grid = torch.cat((xx, yy), dim=1)

    vgrid = grid + flow

    vgrid_x = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=-1)

    output = F.grid_sample(
        x,
        vgrid_norm,
        mode="bilinear",
        padding_mode=PADDING_MODE,
        align_corners=ALIGN_CORNERS,
    )

    if not return_mask:
        return output

    ones = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    valid = F.grid_sample(
        ones,
        vgrid_norm,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=ALIGN_CORNERS,
    )
    valid = (valid > 0.999).type_as(x)
    return output, valid


class RegistrationLoss(nn.Module):
    def __init__(
        self,
        w_epe=1.0,
        w_smooth=0.05,
        w_photo=0.0,
        w_edge=0.05,
        w_ll=0.1,
        w_hf=0.05,
        photo_mode="disabled",
        scale_weights=None,
        smooth_alpha=10.0,
        robust_eps=1e-3,
    ):
        super().__init__()
        self.w_epe = w_epe
        self.w_smooth = w_smooth
        self.w_photo = w_photo
        self.w_edge = w_edge
        self.w_ll = w_ll
        self.w_hf = w_hf

        self.photo_mode = photo_mode
        self.scale_weights = scale_weights
        self.smooth_alpha = smooth_alpha
        self.robust_eps = robust_eps

        self.sobel = Sobelxy()
        self.dwt = HaarDWT2D()

    def _get_scale_weights(self, n_levels):
        if self.scale_weights is not None:
            if len(self.scale_weights) != n_levels:
                raise ValueError(
                    f"scale_weights length mismatch: expected {n_levels}, got {len(self.scale_weights)}"
                )
            return list(self.scale_weights)

        return [0.5 ** (n_levels - 1 - i) for i in range(n_levels)]

    def _prepare_gt_and_mask(self, gt_flow, gt_valid, ref_tensor):
        if gt_flow is None:
            return None, None

        gt_flow = _ensure_flow_2chw(gt_flow).to(ref_tensor.device).type_as(ref_tensor)

        finite_mask = torch.isfinite(gt_flow).all(dim=1, keepdim=True).type_as(ref_tensor)
        gt_flow = torch.nan_to_num(gt_flow, nan=0.0, posinf=0.0, neginf=0.0)

        gt_valid = _ensure_mask(gt_valid, ref_tensor)
        if gt_valid is None:
            gt_valid = finite_mask
        else:
            gt_valid = gt_valid * finite_mask

        return gt_flow, gt_valid

    def epe_loss(self, pred_flows, gt_flow, gt_valid=None):
        pred_flows = _ensure_flow_list(pred_flows)

        if gt_flow is None:
            return pred_flows[-1].new_tensor(0.0)

        gt_flow, gt_valid = self._prepare_gt_and_mask(gt_flow, gt_valid, pred_flows[-1])

        gt_h, gt_w = gt_flow.shape[2:]
        weights = self._get_scale_weights(len(pred_flows))
        weight_sum = sum(weights)

        total_epe = pred_flows[-1].new_tensor(0.0)

        for i, flow in enumerate(pred_flows):
            flow = _ensure_flow_2chw(flow)
            _, _, h, w = flow.shape

            gt_scaled = F.interpolate(gt_flow, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS)
            gt_scaled[:, 0] *= (w / gt_w)
            gt_scaled[:, 1] *= (h / gt_h)

            valid_scaled = None
            if gt_valid is not None:
                valid_scaled = F.interpolate(gt_valid, size=(h, w), mode="nearest")

            diff = flow - gt_scaled
            epe_map = torch.sqrt(torch.sum(diff * diff, dim=1, keepdim=True) + self.robust_eps ** 2)
            loss = masked_mean(epe_map, valid_scaled)
            total_epe += weights[i] * loss

        total_epe = total_epe / weight_sum
        total_epe = torch.nan_to_num(total_epe, nan=0.0, posinf=1e6, neginf=0.0)
        return total_epe

    def smooth_loss(self, flow, guide=None):
        flow = _ensure_flow_2chw(flow)

        dx = image_dx(flow)
        dy = image_dy(flow)

        if guide is not None:
            guide = guide.mean(dim=1, keepdim=True)
            guide_dx = image_dx(guide).abs()
            guide_dy = image_dy(guide).abs()

            wx = torch.exp(-self.smooth_alpha * guide_dx)
            wy = torch.exp(-self.smooth_alpha * guide_dy)

            loss_first = masked_mean(torch.sqrt(dx * dx + self.robust_eps ** 2) * wx) + \
                         masked_mean(torch.sqrt(dy * dy + self.robust_eps ** 2) * wy)
        else:
            loss_first = masked_mean(torch.sqrt(dx * dx + self.robust_eps ** 2)) + \
                         masked_mean(torch.sqrt(dy * dy + self.robust_eps ** 2))

        dxx = image_dx(dx)
        dyy = image_dy(dy)
        loss_second = 0.25 * (
            masked_mean(torch.sqrt(dxx * dxx + self.robust_eps ** 2)) +
            masked_mean(torch.sqrt(dyy * dyy + self.robust_eps ** 2))
        )

        loss = loss_first + loss_second
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
        return loss

    def photo_loss_monomodal(self, warped, target, mask=None):
        l1 = charbonnier_loss(warped, target, eps=self.robust_eps, mask=mask)

        warped_01 = to_01(warped)
        target_01 = to_01(target)

        window = create_window(
            11,
            warped.shape[1],
            device=warped.device,
            dtype=warped.dtype,
        )
        ssim_map = _ssim(warped_01, target_01, window, 11, warped.shape[1], size_average=False)
        if mask is not None:
            l_ssim = 1.0 - masked_mean(ssim_map, mask=mask)
        else:
            l_ssim = 1.0 - ssim_map.mean()

        loss = l1 + l_ssim
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
        return loss

    def cross_modal_edge_loss(self, moving, target, flow):
        warped, valid = warp(moving, flow, return_mask=True)

        edge_warped = self.sobel(warped).mean(dim=1, keepdim=True)
        edge_target = self.sobel(target).mean(dim=1, keepdim=True)

        edge_warped = edge_warped / (masked_mean(edge_warped, valid).detach() + 1e-6)
        edge_target = edge_target / (masked_mean(edge_target, valid).detach() + 1e-6)

        loss = charbonnier_loss(edge_warped, edge_target, eps=self.robust_eps, mask=valid)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
        return loss

    def wavelet_consistency_loss(self, moving, target, flow):
        warped, valid = warp(moving, flow, return_mask=True)

        ll1_w, lh1_w, hl1_w, hh1_w = self.dwt(warped)
        ll1_t, lh1_t, hl1_t, hh1_t = self.dwt(target)

        valid_1 = downsample_mask(valid, times=1)

        ll2_w, _, _, _ = self.dwt(ll1_w)
        ll2_t, _, _, _ = self.dwt(ll1_t)

        valid_2 = downsample_mask(valid, times=2)

        loss_ll = charbonnier_loss(ll2_w, ll2_t, eps=self.robust_eps, mask=valid_2)

        loss_lh = charbonnier_loss(lh1_w, lh1_t, eps=self.robust_eps, mask=valid_1)
        loss_hl = charbonnier_loss(hl1_w, hl1_t, eps=self.robust_eps, mask=valid_1)
        loss_hh = charbonnier_loss(hh1_w, hh1_t, eps=self.robust_eps, mask=valid_1)

        loss_hf = (loss_lh + loss_hl + loss_hh) / 3.0

        loss_ll = torch.nan_to_num(loss_ll, nan=0.0, posinf=1e6, neginf=0.0)
        loss_hf = torch.nan_to_num(loss_hf, nan=0.0, posinf=1e6, neginf=0.0)
        return loss_ll, loss_hf

    def forward(
        self,
        ir_fixed,
        vi_fixed,
        ir_d,
        vi_d,
        flow_preds_ir,
        flow_preds_vi,
        gt_flow_ir=None,
        gt_flow_vi=None,
        gt_valid_ir=None,
        gt_valid_vi=None,
    ):
        flow_preds_ir = _ensure_flow_list(flow_preds_ir)
        flow_preds_vi = _ensure_flow_list(flow_preds_vi)

        loss_epe_ir = self.epe_loss(flow_preds_ir, gt_flow_ir, gt_valid_ir)
        loss_epe_vi = self.epe_loss(flow_preds_vi, gt_flow_vi, gt_valid_vi)
        loss_epe = loss_epe_ir + loss_epe_vi

        loss_smooth = self.smooth_loss(flow_preds_ir[-1], guide=vi_fixed) + \
                      self.smooth_loss(flow_preds_vi[-1], guide=ir_fixed)

        loss_photo = ir_fixed.new_tensor(0.0)
        if self.w_photo > 0:
            if self.photo_mode == "same":
                ir_warped, mask_ir = warp(ir_d, flow_preds_ir[-1], return_mask=True)
                vi_warped, mask_vi = warp(vi_d, flow_preds_vi[-1], return_mask=True)

                loss_photo = self.photo_loss_monomodal(ir_warped, ir_fixed, mask_ir) + \
                             self.photo_loss_monomodal(vi_warped, vi_fixed, mask_vi)

            elif self.photo_mode == "disabled":
                loss_photo = ir_fixed.new_tensor(0.0)
            else:
                raise NotImplementedError(
                    f"photo_mode={self.photo_mode} is not supported. "
                    f"For current cross-modal branches, keep photo_mode='disabled' and reg_photo=0.0."
                )

        loss_edge = ir_fixed.new_tensor(0.0)
        if self.w_edge > 0:
            loss_edge = self.cross_modal_edge_loss(ir_d, vi_fixed, flow_preds_ir[-1]) + \
                        self.cross_modal_edge_loss(vi_d, ir_fixed, flow_preds_vi[-1])

        loss_ll_ir, loss_hf_ir = self.wavelet_consistency_loss(ir_d, vi_fixed, flow_preds_ir[-1])
        loss_ll_vi, loss_hf_vi = self.wavelet_consistency_loss(vi_d, ir_fixed, flow_preds_vi[-1])

        loss_ll = loss_ll_ir + loss_ll_vi
        loss_hf = loss_hf_ir + loss_hf_vi

        total = (
            self.w_epe * loss_epe +
            self.w_smooth * loss_smooth +
            self.w_photo * loss_photo +
            self.w_edge * loss_edge +
            self.w_ll * loss_ll +
            self.w_hf * loss_hf
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)

        return total, loss_epe, loss_smooth, loss_photo, loss_edge, loss_ll, loss_hf


class FusionRegistrationLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        default_config = {
            "fusion_grad": 10.0,
            "fusion_int": 1.0,
            "fusion_ssim": 5.0,
            "fusion_mean": 1.0,
            "fusion_contrast": 1.0,
            "reg_epe": 1.0,
            "reg_smooth": 0.05,
            "reg_photo": 0.0,
            "reg_edge": 0.05,
            "reg_ll": 0.10,
            "reg_hf": 0.05,
            "photo_mode": "disabled",
            "reg_scale_weights": None,
            "weight_reg_vs_fusion": 0.2,
        }

        if config is not None:
            default_config.update(config)
        config = default_config

        self.fusion_crit = Fusion_loss(
            w_grad=config["fusion_grad"],
            w_int=config["fusion_int"],
            w_ssim=config["fusion_ssim"],
            w_mean=config["fusion_mean"],
            w_contrast=config["fusion_contrast"],
        )

        self.reg_crit = RegistrationLoss(
            w_epe=config["reg_epe"],
            w_smooth=config["reg_smooth"],
            w_photo=config["reg_photo"],
            w_edge=config["reg_edge"],
            w_ll=config["reg_ll"],
            w_hf=config["reg_hf"],
            photo_mode=config["photo_mode"],
            scale_weights=config["reg_scale_weights"],
        )

        self.reg_weight = config["weight_reg_vs_fusion"]

    def forward(
        self,
        img_vis_fixed,
        img_ir_fixed,
        img_fused,
        ir_fixed,
        vi_fixed,
        ir_d,
        vi_d,
        flow_preds_ir,
        flow_preds_vi,
        gt_flow_ir=None,
        gt_flow_vi=None,
        gt_valid_ir=None,
        gt_valid_vi=None,
    ):
        l_fusion, l_grad, l_int, l_ssim, l_mean, l_contrast = self.fusion_crit(
            img_vis_fixed,
            img_ir_fixed,
            img_fused,
        )

        l_reg, l_epe, l_smooth, l_photo, l_edge, l_ll, l_hf = self.reg_crit(
            ir_fixed,
            vi_fixed,
            ir_d,
            vi_d,
            flow_preds_ir,
            flow_preds_vi,
            gt_flow_ir,
            gt_flow_vi,
            gt_valid_ir,
            gt_valid_vi,
        )

        total_loss = l_fusion + self.reg_weight * l_reg
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=0.0)

        log_dict = {
            "Total": total_loss,
            "Fusion/Total": l_fusion,
            "Fusion/Grad": l_grad,
            "Fusion/Int": l_int,
            "Fusion/SSIM": l_ssim,
            "Fusion/Mean": l_mean,
            "Fusion/Contrast": l_contrast,
            "Reg/Total": l_reg,
            "Reg/EPE": l_epe,
            "Reg/Smooth": l_smooth,
            "Reg/Photo": l_photo,
            "Reg/Edge": l_edge,
            "Reg/LL": l_ll,
            "Reg/HF": l_hf,
        }

        return total_loss, log_dict
