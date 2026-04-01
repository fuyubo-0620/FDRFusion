import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class IRVIFusionRegistrationDataset(Dataset):
    def __init__(self,
                 root_dir,
                 img_size=(256, 256),
                 ir_dir='ir',
                 vi_dir='vi',
                 ir_d_dir='ir_d',
                 vi_d_dir='vi_d',
                 ir_flow_dir='ir_flows',
                 vi_flow_dir='vi_flows',
                 ir_valid_dir='ir_valid',
                 vi_valid_dir='vi_valid',
                 strict_flow=True):
        super().__init__()

        self.root_dir = root_dir
        self.img_size = tuple(img_size)
        self.strict_flow = strict_flow

        self.paths = {
            'ir': os.path.join(root_dir, ir_dir),
            'vi': os.path.join(root_dir, vi_dir),
            'ir_d': os.path.join(root_dir, ir_d_dir),
            'vi_d': os.path.join(root_dir, vi_d_dir),
            'ir_flow': os.path.join(root_dir, ir_flow_dir),
            'vi_flow': os.path.join(root_dir, vi_flow_dir),
            'ir_valid': os.path.join(root_dir, ir_valid_dir),
            'vi_valid': os.path.join(root_dir, vi_valid_dir),
        }

        for k in ['ir', 'vi', 'ir_d', 'vi_d', 'ir_flow', 'vi_flow']:
            if not os.path.isdir(self.paths[k]):
                raise FileNotFoundError(f"Missing directory: {self.paths[k]}")

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        ir_files = sorted([
            f for f in os.listdir(self.paths['ir'])
            if f.lower().endswith(valid_exts)
        ])

        if len(ir_files) == 0:
            raise ValueError(f"No images found in {self.paths['ir']}")

        file_list = []
        for fname in ir_files:
            stem = os.path.splitext(fname)[0]
            filename_npy = stem + '.npy'

            ok = True
            required_files = [
                os.path.join(self.paths['vi'], fname),
                os.path.join(self.paths['ir_d'], fname),
                os.path.join(self.paths['vi_d'], fname),
                os.path.join(self.paths['ir_flow'], filename_npy),
                os.path.join(self.paths['vi_flow'], filename_npy),
            ]
            for p in required_files:
                if not os.path.exists(p):
                    ok = False
                    break

            if ok:
                file_list.append(fname)

        if len(file_list) == 0:
            raise ValueError("No valid paired samples found. Please check dataset generation.")

        self.file_list = file_list
        print(f"[Dataset] Loaded {len(self.file_list)} valid samples from {root_dir}")

    def __len__(self):
        return len(self.file_list)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        if (img.shape[0], img.shape[1]) != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img).unsqueeze(0)

    def _normalize_flow_shape(self, flow_np):
        if flow_np.ndim != 3:
            raise ValueError(f"Flow must be 3D, got shape {flow_np.shape}")

        if flow_np.shape[0] == 2:
            return flow_np.astype(np.float32)

        if flow_np.shape[-1] == 2:
            return np.transpose(flow_np, (2, 0, 1)).astype(np.float32)

        raise ValueError(f"Unsupported flow shape: {flow_np.shape}")

    def _resize_flow(self, flow_t):
        _, h, w = flow_t.shape
        th, tw = self.img_size

        if (h, w) == (th, tw):
            return flow_t

        flow_t = flow_t.unsqueeze(0)
        flow_t = F.interpolate(flow_t, size=(th, tw), mode='bilinear', align_corners=True)

        scale_x = float(tw) / float(w)
        scale_y = float(th) / float(h)
        flow_t[:, 0] *= scale_x
        flow_t[:, 1] *= scale_y
        return flow_t.squeeze(0)

    def _load_flow(self, path):
        if not os.path.exists(path):
            if self.strict_flow:
                raise FileNotFoundError(f"Missing flow file: {path}")
            return torch.zeros(2, *self.img_size, dtype=torch.float32)

        flow_np = np.load(path)
        flow_np = self._normalize_flow_shape(flow_np)
        flow_t = torch.from_numpy(flow_np).float()
        flow_t = self._resize_flow(flow_t)
        return flow_t

    def _load_mask(self, path):
        if not os.path.exists(path):
            return torch.ones(1, *self.img_size, dtype=torch.float32)

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return torch.ones(1, *self.img_size, dtype=torch.float32)

        if (mask.shape[0], mask.shape[1]) != self.img_size:
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        mask = (mask.astype(np.float32) > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        stem = os.path.splitext(filename)[0]
        filename_npy = stem + '.npy'

        p_ir = os.path.join(self.paths['ir'], filename)
        p_vi = os.path.join(self.paths['vi'], filename)
        p_ir_d = os.path.join(self.paths['ir_d'], filename)
        p_vi_d = os.path.join(self.paths['vi_d'], filename)

        p_ir_flow = os.path.join(self.paths['ir_flow'], filename_npy)
        p_vi_flow = os.path.join(self.paths['vi_flow'], filename_npy)

        p_ir_valid = os.path.join(self.paths['ir_valid'], filename)
        p_vi_valid = os.path.join(self.paths['vi_valid'], filename)

        ir = self._load_img(p_ir)
        vi = self._load_img(p_vi)
        ir_d = self._load_img(p_ir_d)
        vi_d = self._load_img(p_vi_d)

        ir_flow = self._load_flow(p_ir_flow)
        vi_flow = self._load_flow(p_vi_flow)

        ir_valid = self._load_mask(p_ir_valid)
        vi_valid = self._load_mask(p_vi_valid)

        return {
            'name': filename,
            'ir': ir,
            'vi': vi,
            'ir_d': ir_d,
            'vi_d': vi_d,
            'ir_flow': ir_flow,
            'vi_flow': vi_flow,
            'ir_valid': ir_valid,
            'vi_valid': vi_valid
        }

if __name__ == '__main__':
    root = r"./data/msrs_train"
    if os.path.exists(root):
        ds = IRVIFusionRegistrationDataset(root)
        sample = ds[0]

        print(f"Sample: {sample['name']}")
        print(f"IR shape: {sample['ir'].shape}")
        print(f"IR range: {sample['ir'].min():.3f} ~ {sample['ir'].max():.3f}")
        print(f"IR flow shape: {sample['ir_flow'].shape}")
        print(f"IR flow range: {sample['ir_flow'].min():.3f} ~ {sample['ir_flow'].max():.3f}")
        print(f"IR valid ratio: {sample['ir_valid'].mean():.3f}")
    else:
        print("Dataset root does not exist.")