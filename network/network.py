import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import kornia.filters as KF
import kornia.utils as KU
from timm.layers import DropPath, to_2tuple, trunc_normal_

ALIGN_CORNERS = True
FLOW_PADDING_MODE = "reflection"
FLOW_SCALE = 1.0
MAX_DISP_PX = 8.0


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape

        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, n):
        flops = 0
        flops += n * self.dim * 3 * self.dim
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


class Cross_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        b_, n, c = x.shape

        q = self.q(x).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(b_, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = q[0], kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, n):
        flops = 0
        flops += n * self.dim * 3 * self.dim
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, l, c = x.shape

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        n_w = h * w / self.window_size / self.window_size
        flops += n_w * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * h * w
        return flops


class Cross_SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)

        self.attn_A = Cross_WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.attn_B = Cross_WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path_A = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, x_size):
        h, w = x_size
        b, l, c = x.shape

        shortcut_A = x
        shortcut_B = y

        x = self.norm1_A(x).view(b, h, w, c)
        y = self.norm1_B(y).view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)
        y_windows = window_partition(shifted_y, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)
        else:
            mask = self.calculate_mask(x_size).to(x.device)
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=mask)
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=mask)

        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, c)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, c)

        shifted_x = window_reverse(attn_windows_A, self.window_size, h, w)
        shifted_y = window_reverse(attn_windows_B, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y

        x = x.view(b, h * w, c)
        y = y.view(b, h * w, c)

        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))

        return x, y

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w * 2
        n_w = h * w / self.window_size / self.window_size
        flops += n_w * self.attn_A.flops(self.window_size * self.window_size) * 2
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio * 2
        flops += self.dim * h * w * 2
        return flops


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class Cross_BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                Cross_SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, y, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, y = checkpoint.checkpoint(blk, x, y, x_size, use_reentrant=False)
            else:
                x, y = blk(x, y, x_size)

        if self.downsample is not None:
            x = self.downsample(x)
            y = self.downsample(y)

        return x, y

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class CRSTB(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        args = dict(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )
        self.residual_group = Cross_BasicLayer(**args)
        self.residual_group_A = BasicLayer(**args)
        self.residual_group_B = BasicLayer(**args)

    def forward(self, x, y, x_size):
        x = self.residual_group_A(x, x_size)
        y = self.residual_group_B(y, x_size)
        x, y = self.residual_group(x, y, x_size)
        return x, y

    def flops(self):
        return self.residual_group_A.flops() + self.residual_group_B.flops() + self.residual_group.flops()


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm=True):
        super().__init__()
        self.conv3x3 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not norm,
        )
        self.dilated_conv3x3 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            bias=not norm,
        )
        self.norm = norm
        if self.norm:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(0.1, inplace=False)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.dilated_conv3x3(x)
        x_cat = torch.cat([x1, x2], dim=1)
        if self.norm:
            x_cat = self.bn(x_cat)
        x_cat = self.relu(x_cat)
        return self.ca(x_cat)


class FeatureExtractor(nn.Module):
    def __init__(self, ic, oc, depth, norm=True):
        super().__init__()
        self.depth = depth
        self.ms_branches = nn.ModuleList()
        self.res_connections = nn.ModuleList()

        for i in range(depth):
            curr_in_ch = ic if i == 0 else oc
            self.ms_branches.append(MultiScaleBranch(curr_in_ch, oc, stride=1, norm=norm))
            self.res_connections.append(nn.Conv2d(curr_in_ch, oc, kernel_size=1, stride=1, bias=not norm))

        self.final_norm = nn.InstanceNorm2d(oc, affine=True) if norm else nn.Identity()

    def forward(self, x):
        for i in range(self.depth):
            ms_feat = self.ms_branches[i](x)
            res_feat = self.res_connections[i](x)
            x = ms_feat + res_feat
        return self.final_norm(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        return self.num_patches * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.embed_dim = embed_dim

    def forward(self, x, x_size=None):
        b, hw, c = x.shape
        if x_size is None:
            x_size = self.patches_resolution
        h_patch, w_patch = x_size
        x = x.transpose(1, 2).view(b, self.embed_dim, h_patch, w_patch)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, norm="IN", act=True):
        super().__init__()
        use_bias = (norm is None) or (norm == "Tanh")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=use_bias,
        )

        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "Tanh":
            self.norm = nn.Tanh()
            act = False
        else:
            self.norm = None

        self.act = nn.LeakyReLU(0.2, inplace=False) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


def pixel_flow_to_norm(flow, h, w, align_corners=True):
    flow_norm = flow.clone()

    if flow.dim() != 4:
        raise ValueError(f"flow must be 4D, but got shape {flow.shape}")

    if flow.shape[1] == 2:
        if align_corners:
            flow_norm[:, 0] = flow_norm[:, 0] * 2.0 / max(w - 1, 1)
            flow_norm[:, 1] = flow_norm[:, 1] * 2.0 / max(h - 1, 1)
        else:
            flow_norm[:, 0] = flow_norm[:, 0] * 2.0 / max(w, 1)
            flow_norm[:, 1] = flow_norm[:, 1] * 2.0 / max(h, 1)
    elif flow.shape[-1] == 2:
        if align_corners:
            flow_norm[..., 0] = flow_norm[..., 0] * 2.0 / max(w - 1, 1)
            flow_norm[..., 1] = flow_norm[..., 1] * 2.0 / max(h - 1, 1)
        else:
            flow_norm[..., 0] = flow_norm[..., 0] * 2.0 / max(w, 1)
            flow_norm[..., 1] = flow_norm[..., 1] * 2.0 / max(h, 1)
    else:
        raise ValueError(f"Unsupported flow shape: {flow.shape}")

    return flow_norm


def flow_warp(x, flow, mode="bilinear", padding_mode="reflection", align_corners=True, flow_in_pixel=False):
    b, c, h, w = x.shape

    if flow_in_pixel:
        flow = pixel_flow_to_norm(flow, h, w, align_corners=align_corners)

    if flow.shape[1] == 2:
        flow = flow.permute(0, 2, 3, 1).contiguous()

    grid = KU.create_meshgrid(h, w, normalized_coordinates=True, device=x.device, dtype=x.dtype)
    grid = grid.expand(b, -1, -1, -1).contiguous()

    vgrid = grid + flow

    output = F.grid_sample(
        x,
        vgrid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return output


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


class DispEstimator(nn.Module):
    def __init__(
        self,
        channel,
        depth=4,
        norm="IN",
        feat_dilation=1,
        corrks=7,
        corr_dilation=1,
        flow_scale=FLOW_SCALE,
    ):
        super().__init__()

        self.corrks = corrks
        self.corr_dilation = corr_dilation
        self.flow_scale = float(flow_scale)
        self.eps = 1e-6

        self.preprocessor = Conv2d(
            channel,
            channel,
            3,
            padding=feat_dilation,
            dilation=feat_dilation,
            norm=None,
            act=False,
        )

        self.featcompressor = nn.Sequential(
            Conv2d(channel * 2, channel * 2, 3, padding=1, norm=norm, act=True),
            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=False),
        )

        current_ic = channel + corrks * corrks
        current_oc = channel
        layers = []
        curr_dil = 1

        for _ in range(depth - 1):
            next_oc = max(16, current_oc // 2)
            layers.append(
                Conv2d(
                    current_ic,
                    current_oc,
                    3,
                    stride=1,
                    padding=curr_dil,
                    dilation=curr_dil,
                    norm=norm,
                    act=True,
                )
            )
            current_ic = current_oc
            current_oc = next_oc
            curr_dil *= 2

        self.estimator = nn.Sequential(*layers)
        self.flow_head = nn.Conv2d(current_ic, 2, kernel_size=3, padding=1)

        nn.init.constant_(self.flow_head.weight, 0.0)
        nn.init.constant_(self.flow_head.bias, 0.0)

    def localcorr(self, feat1, feat2):
        feat = self.featcompressor(torch.cat([feat1, feat2], dim=1))

        b, c, h, w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1, (7, 7), (1.5, 1.5), border_type="replicate")

        pad = self.corr_dilation * (self.corrks // 2)
        feat1_loc_blk = F.unfold(
            feat1_smooth,
            kernel_size=self.corrks,
            dilation=self.corr_dilation,
            padding=pad,
            stride=1,
        ).reshape(b, c, self.corrks * self.corrks, h, w)

        localcorr = (feat2.unsqueeze(2) - feat1_loc_blk).pow(2).mean(dim=1)
        localcorr_mean = localcorr.mean(dim=(1, 2, 3), keepdim=True)
        localcorr = localcorr / (localcorr_mean + self.eps)

        corr = torch.cat([feat, localcorr], dim=1)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr

    def forward(self, feat1, feat2):
        b = feat1.shape[0]

        feat = torch.cat([feat1, feat2], dim=0)
        feat = self.preprocessor(feat)
        feat1, feat2 = feat[:b], feat[b:]

        corr = self.localcorr(feat1, feat2)
        x = self.estimator(corr)
        disp = self.flow_head(x) * self.flow_scale

        disp = KF.gaussian_blur2d(disp, (7, 7), (1.5, 1.5), border_type="replicate")
        disp = torch.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        return disp


class DispRefiner(nn.Module):
    def __init__(self, channel, dilation=1, depth=4, norm="IN"):
        super().__init__()

        self.preprocessor = nn.Sequential(
            Conv2d(channel, channel, 3, dilation=dilation, padding=dilation, norm=None, act=False)
        )

        self.featcompressor = nn.Sequential(
            Conv2d(channel * 2, channel * 2, 3, padding=1, norm=norm, act=True),
            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=False),
        )

        current_ic = channel + 2
        current_oc = channel
        layers = []
        curr_dil = 1

        for _ in range(depth - 1):
            next_oc = max(16, current_oc // 2)
            layers.append(
                Conv2d(
                    current_ic,
                    current_oc,
                    kernel_size=3,
                    stride=1,
                    padding=curr_dil,
                    dilation=curr_dil,
                    norm=norm,
                    act=True,
                )
            )
            current_ic = current_oc
            current_oc = next_oc
            curr_dil *= 2

        self.estimator = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(current_ic, 2, kernel_size=3, padding=1)

        nn.init.constant_(self.out_conv.weight, 0.0)
        nn.init.constant_(self.out_conv.bias, 0.0)

    def forward(self, feat1, feat2, disp):
        b = feat1.shape[0]

        feat = torch.cat([feat1, feat2], dim=0)
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b], feat[b:]], dim=1))

        x = torch.cat([feat, disp], dim=1)
        delta_disp = self.out_conv(self.estimator(x))
        disp = disp + delta_disp
        disp = torch.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        return disp


class WaveletHFRefiner(nn.Module):
    def __init__(self, low_ch, high_ch, depth=4, norm="IN"):
        super().__init__()

        self.low_fuse = nn.Sequential(
            Conv2d(low_ch * 2, low_ch, 3, padding=1, norm=norm, act=True),
            Conv2d(low_ch, low_ch, 3, padding=1, norm=None, act=False),
        )

        self.high_fuse = nn.Sequential(
            Conv2d(high_ch * 2, high_ch, 3, padding=1, norm=norm, act=True),
            Conv2d(high_ch, high_ch, 3, padding=1, norm=None, act=False),
        )

        self.gate = nn.Sequential(
            Conv2d(low_ch + high_ch, high_ch, 3, padding=1, norm=None, act=False),
            nn.Sigmoid(),
        )

        current_ic = low_ch + high_ch + 2
        current_oc = max(32, low_ch)
        layers = []
        curr_dil = 1

        for _ in range(depth - 1):
            next_oc = max(16, current_oc // 2)
            layers.append(
                Conv2d(
                    current_ic,
                    current_oc,
                    3,
                    stride=1,
                    padding=curr_dil,
                    dilation=curr_dil,
                    norm=norm,
                    act=True,
                )
            )
            current_ic = current_oc
            current_oc = next_oc
            curr_dil *= 2

        self.estimator = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(current_ic, 2, kernel_size=3, padding=1)

        nn.init.constant_(self.out_conv.weight, 0.0)
        nn.init.constant_(self.out_conv.bias, 0.0)

    def forward(self, low_src_warp, low_tgt, high_src_warp, high_tgt, disp):
        low_feat = self.low_fuse(torch.cat([low_src_warp, low_tgt], dim=1))
        high_feat = self.high_fuse(torch.cat([high_src_warp, high_tgt], dim=1))

        gate = self.gate(torch.cat([low_feat, high_feat], dim=1))
        x = torch.cat([low_feat, high_feat * gate, disp], dim=1)

        delta = self.out_conv(self.estimator(x))
        disp = disp + delta
        disp = torch.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        return disp


class DenseMatcher(nn.Module):
    def __init__(self, in_channels=16, flow_scale=FLOW_SCALE, max_disp=MAX_DISP_PX):
        super().__init__()
        self.flow_scale = float(flow_scale)
        self.max_disp = float(max_disp)

        base_oc = 32

        self.share1 = Conv2d(in_channels, base_oc, 3, padding=1, norm="IN")

        self.dwt1 = HaarDWT2D()
        self.dwt2 = HaarDWT2D()

        self.ll1_encoder = nn.Sequential(
            Conv2d(base_oc, base_oc * 2, 3, padding=1, norm="IN"),
            Conv2d(base_oc * 2, base_oc * 2, 3, padding=1, norm="IN"),
        )

        self.hf1_encoder = nn.Sequential(
            Conv2d(base_oc * 3, base_oc * 2, 3, padding=1, norm="IN"),
            Conv2d(base_oc * 2, base_oc * 2, 3, padding=1, norm="IN"),
        )

        self.ll2_smoother = nn.Sequential(
            Conv2d(base_oc * 2, base_oc * 2, 3, padding=1, norm="IN"),
            Conv2d(base_oc * 2, base_oc * 2, 3, padding=1, norm="IN"),
        )

        self.matcher_coarse = DispEstimator(
            channel=base_oc * 2,
            depth=4,
            norm="IN",
            feat_dilation=2,
            corrks=7,
            corr_dilation=2,
            flow_scale=self.flow_scale,
        )

        self.matcher_mid = DispEstimator(
            channel=base_oc * 2,
            depth=4,
            norm="IN",
            feat_dilation=2,
            corrks=7,
            corr_dilation=2,
            flow_scale=self.flow_scale,
        )

        self.hf_refiner = WaveletHFRefiner(low_ch=base_oc * 2, high_ch=base_oc * 2, depth=4, norm="IN")

        self.final_refiner = DispRefiner(channel=base_oc, dilation=1, depth=4, norm="IN")

    def _px2norm(self, disp_px, h, w):
        return pixel_flow_to_norm(disp_px, h, w, align_corners=ALIGN_CORNERS)

    def _upsample_flow(self, flow, target_size):
        h_in, w_in = flow.shape[2:]
        h_out, w_out = target_size

        if (h_in == h_out) and (w_in == w_out):
            return flow

        scale_h = float(h_out) / float(h_in)
        scale_w = float(w_out) / float(w_in)

        flow_up = F.interpolate(flow, size=target_size, mode="bilinear", align_corners=ALIGN_CORNERS)
        flow_up[:, 0] *= scale_w
        flow_up[:, 1] *= scale_h
        return flow_up

    def forward(self, src, tgt):
        b = src.shape[0]

        feat0 = torch.cat([src, tgt], dim=0)
        feat1 = self.share1(feat0)
        feat1_src, feat1_tgt = feat1[:b], feat1[b:]

        ll1_src_raw, lh1_src, hl1_src, hh1_src = self.dwt1(feat1_src)
        ll1_tgt_raw, lh1_tgt, hl1_tgt, hh1_tgt = self.dwt1(feat1_tgt)

        ll1_src = self.ll1_encoder(ll1_src_raw)
        ll1_tgt = self.ll1_encoder(ll1_tgt_raw)

        hf1_src = self.hf1_encoder(torch.cat([lh1_src, hl1_src, hh1_src], dim=1))
        hf1_tgt = self.hf1_encoder(torch.cat([lh1_tgt, hl1_tgt, hh1_tgt], dim=1))

        ll2_src, _, _, _ = self.dwt2(ll1_src)
        ll2_tgt, _, _, _ = self.dwt2(ll1_tgt)

        ll2_src = self.ll2_smoother(ll2_src)
        ll2_tgt = self.ll2_smoother(ll2_tgt)

        disp3_px = self.matcher_coarse(ll2_src, ll2_tgt)
        disp3_px = torch.clamp(disp3_px, -self.max_disp / 4.0, self.max_disp / 4.0)

        h2, w2 = ll1_src.shape[2:]
        disp2_init_px = self._upsample_flow(disp3_px, (h2, w2))

        ll1_src_warped = flow_warp(
            ll1_src,
            disp2_init_px,
            padding_mode=FLOW_PADDING_MODE,
            align_corners=ALIGN_CORNERS,
            flow_in_pixel=True,
        )

        disp2_res_px = self.matcher_mid(ll1_src_warped, ll1_tgt)
        disp2_total_px = disp2_init_px + disp2_res_px
        disp2_total_px = torch.clamp(disp2_total_px, -self.max_disp / 2.0, self.max_disp / 2.0)

        ll1_src_warped = flow_warp(
            ll1_src,
            disp2_total_px,
            padding_mode=FLOW_PADDING_MODE,
            align_corners=ALIGN_CORNERS,
            flow_in_pixel=True,
        )

        hf1_src_warped = flow_warp(
            hf1_src,
            disp2_total_px,
            padding_mode=FLOW_PADDING_MODE,
            align_corners=ALIGN_CORNERS,
            flow_in_pixel=True,
        )

        disp2_hf_px = self.hf_refiner(ll1_src_warped, ll1_tgt, hf1_src_warped, hf1_tgt, disp2_total_px)
        disp2_hf_px = torch.clamp(disp2_hf_px, -self.max_disp / 2.0, self.max_disp / 2.0)

        h1, w1 = feat1_src.shape[2:]
        disp1_init_px = self._upsample_flow(disp2_hf_px, (h1, w1))

        feat1_src_warped = flow_warp(
            feat1_src,
            disp1_init_px,
            padding_mode=FLOW_PADDING_MODE,
            align_corners=ALIGN_CORNERS,
            flow_in_pixel=True,
        )

        final_disp_px = self.final_refiner(feat1_src_warped, feat1_tgt, disp1_init_px)

        final_disp_px = torch.clamp(final_disp_px, -self.max_disp, self.max_disp)
        final_disp_px = KF.gaussian_blur2d(final_disp_px, (9, 9), (1.5, 1.5), border_type="replicate")
        final_disp_px = torch.nan_to_num(final_disp_px, nan=0.0, posinf=0.0, neginf=0.0)

        coarse_disp_px = self._upsample_flow(disp3_px, (h1, w1))
        mid_disp_px = self._upsample_flow(disp2_hf_px, (h1, w1))

        coarse_disp_norm = self._px2norm(coarse_disp_px, h1, w1)
        mid_disp_norm = self._px2norm(mid_disp_px, h1, w1)
        final_disp_norm = self._px2norm(final_disp_px, h1, w1)

        output = {
            "disp": final_disp_px,
            "disp_norm": final_disp_norm,
            "flow_preds": [coarse_disp_px, mid_disp_px, final_disp_px],
            "flow_preds_norm": [coarse_disp_norm, mid_disp_norm, final_disp_norm],
        }
        return output


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=True,
        )
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class ReconNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = FusionConv(in_ch, 2 * in_ch, 3, stride=1, padding=1, act=True)
        self.conv2 = FusionConv(2 * in_ch, 4 * in_ch, 3, stride=1, padding=1, act=True)
        self.conv3 = FusionConv(4 * in_ch, 8 * in_ch, 3, stride=1, padding=1, act=True)
        self.proj = FusionConv(in_ch, 8 * in_ch, 3, stride=1, padding=1, act=False)

    def forward(self, x):
        res = x
        x = self.conv3(self.conv2(self.conv1(x)))
        out = x + self.proj(res)
        return out


class FusionReconNet(nn.Module):
    def __init__(self, in_ch, detail_scale=0.1):
        super().__init__()
        self.detail_scale = detail_scale

        c1 = max(32, in_ch // 4)
        c2 = max(16, in_ch // 8)

        self.stem = nn.Sequential(
            FusionConv(in_ch, c1, 3, 1, 1, act=True),
            FusionConv(c1, c1, 3, 1, 1, act=True),
            FusionConv(c1, c2, 3, 1, 1, act=True),
        )

        self.alpha_head = nn.Sequential(
            nn.Conv2d(c2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.detail_head = nn.Sequential(
            nn.Conv2d(c2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

        nn.init.constant_(self.alpha_head[0].weight, 0.0)
        nn.init.constant_(self.alpha_head[0].bias, 0.0)
        nn.init.constant_(self.detail_head[0].weight, 0.0)
        nn.init.constant_(self.detail_head[0].bias, 0.0)

    def forward(self, feat, ir_img, vi_img):
        x = self.stem(feat)
        alpha = self.alpha_head(x)
        detail = self.detail_scale * self.detail_head(x)

        fused_base = alpha * vi_img + (1.0 - alpha) * ir_img
        fused = (fused_base + detail).clamp(-1.0, 1.0)

        return fused, alpha, detail


class FusionNet(nn.Module):
    def __init__(self, in_ch, up_scale=None, up_mode="bilinear", detail_scale=0.1):
        super().__init__()
        self.recon = ReconNet(in_ch)
        self.fusion_recon = FusionReconNet(in_ch * 8, detail_scale=detail_scale)
        self.up_scale = up_scale
        self.up_mode = up_mode

    def forward(self, x, ir_img, vi_img):
        feat = self.recon(x)

        if self.up_scale is not None and self.up_scale != 1:
            feat = F.interpolate(
                feat,
                scale_factor=self.up_scale,
                mode=self.up_mode,
                align_corners=False if self.up_mode in ["bilinear", "bicubic"] else None,
            )

        fused, alpha, detail = self.fusion_recon(feat, ir_img, vi_img)
        return fused, alpha, detail


class WaveLetFusion(nn.Module):
    def __init__(
        self,
        image_size=256,
        in_ch=1,
        feat_dim=16,
        patch_size=4,
        embed_dim=96,
        num_heads=6,
        window_size=4,
        depth=1,
        drop_path_rate=0.0,
        flow_scale=FLOW_SCALE,
        max_disp=MAX_DISP_PX,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        self.DM_ir = DenseMatcher(in_channels=feat_dim, flow_scale=flow_scale, max_disp=max_disp)
        self.DM_vi = DenseMatcher(in_channels=feat_dim, flow_scale=flow_scale, max_disp=max_disp)

        self.fe_vi = FeatureExtractor(ic=in_ch, oc=feat_dim, depth=2, norm=True)
        self.fe_ir = FeatureExtractor(ic=in_ch, oc=feat_dim, depth=2, norm=True)

        embed_in_ch = feat_dim * 2
        self.embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=embed_in_ch,
            embed_dim=embed_dim,
        )
        self.unembed = PatchUnEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=embed_in_ch,
            embed_dim=embed_dim,
        )

        self.feature_resolution = (image_size // patch_size, image_size // patch_size)
        self.feature_attention = CRSTB(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            input_resolution=self.feature_resolution,
        )

        self.FN = FusionNet(in_ch=embed_dim * 2, up_scale=patch_size, detail_scale=0.1)

    def forward(self, ir, vi, ir_d, vi_d):
        ir_1 = self.fe_ir(ir)
        ir_d_1 = self.fe_ir(ir_d)
        vi_1 = self.fe_vi(vi)
        vi_d_1 = self.fe_vi(vi_d)

        out_ir = self.DM_ir(src=ir_d_1, tgt=vi_1)
        disp_ir_px = out_ir["disp"]
        flow_preds_ir = out_ir["flow_preds"]

        ir_reg = flow_warp(
            ir_d_1,
            disp_ir_px,
            padding_mode=FLOW_PADDING_MODE,
            align_corners=ALIGN_CORNERS,
            flow_in_pixel=True,
        )

        out_vi = self.DM_vi(src=vi_d_1, tgt=ir_1)
        disp_vi_px = out_vi["disp"]
        flow_preds_vi = out_vi["flow_preds"]

        vi_reg = flow_warp(
            vi_d_1,
            disp_vi_px,
            padding_mode=FLOW_PADDING_MODE,
            align_corners=ALIGN_CORNERS,
            flow_in_pixel=True,
        )

        ir_f = torch.cat([ir_1, ir_reg], dim=1)
        vi_f = torch.cat([vi_1, vi_reg], dim=1)

        ir_embed = self.embed(ir_f)
        vi_embed = self.embed(vi_f)

        ir_vi, vi_ir = self.feature_attention(ir_embed, vi_embed, self.feature_resolution)

        ir_vi_img = self.unembed(ir_vi)
        vi_ir_img = self.unembed(vi_ir)

        fusion_feat = torch.cat([ir_vi_img, vi_ir_img], dim=1)
        fused, alpha, detail = self.FN(fusion_feat, ir, vi)

        return fused, flow_preds_ir, flow_preds_vi
