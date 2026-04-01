import numpy as np
import cv2
import os
import random

def generate_local_flow(h, w, num_points=3, sigma=50, magnitude=10):
    flow = np.zeros((h, w, 2), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    for _ in range(num_points):
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)

        angle = np.random.uniform(0, 2 * np.pi)
        strength = np.random.uniform(0.5, 1.0) * magnitude

        dx_val = strength * np.cos(angle)
        dy_val = strength * np.sin(angle)

        dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        weight = np.exp(-dist_sq / (2 * sigma ** 2))

        flow[..., 0] += weight * dx_val
        flow[..., 1] += weight * dy_val

    return flow

def apply_deformation(img, flow):
    h, w = img.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    flow = flow.astype(np.float32)
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    deformed_img = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return deformed_img

def main():
    ir_dir = r'/tmp/pycharm_project_755/data/deformated/ir'
    vi_dir = r'/tmp/pycharm_project_755/data/deformated/vi'

    output_root = r'/tmp/pycharm_project_755/data/deformated'

    ir_deformed_dir = os.path.join(output_root, 'ir_d')
    vi_deformed_dir = os.path.join(output_root, 'vi_d')

    ir_flow_dir = os.path.join(output_root, 'ir_flows')
    vi_flow_dir = os.path.join(output_root, 'vi_flows')

    os.makedirs(ir_deformed_dir, exist_ok=True)
    os.makedirs(vi_deformed_dir, exist_ok=True)
    os.makedirs(ir_flow_dir, exist_ok=True)
    os.makedirs(vi_flow_dir, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(img_exts)])

    print(f"Found {len(ir_files)} IR images in {ir_dir}")

    for idx, fname in enumerate(ir_files):
        ir_path = os.path.join(ir_dir, fname)
        vi_path = os.path.join(vi_dir, fname)

        if not os.path.exists(vi_path):
            print(f"[Skip] No corresponding VI image: {vi_path}")
            continue

        ir = cv2.imread(ir_path, 0)
        vi = cv2.imread(vi_path, 0)

        if ir is None or vi is None:
            print(f"[Skip] Failed to read: {ir_path} or {vi_path}")
            continue

        if ir.shape != vi.shape:
            h, w = ir.shape[:2]
            vi = cv2.resize(vi, (w, h))
        else:
            h, w = ir.shape[:2]

        print(f"[{idx+1}/{len(ir_files)}] Processing {fname} | Size: ({h}, {w})")

        flow_ir = generate_local_flow(h, w, num_points=2, sigma=100, magnitude=15)
        flow_vi = generate_local_flow(h, w, num_points=2, sigma=100, magnitude=15)

        ir_d = apply_deformation(ir, flow_ir)
        vi_d = apply_deformation(vi, flow_vi)

        stem, _ = os.path.splitext(fname)

        ir_d_path = os.path.join(ir_deformed_dir, f"{stem}.png")
        vi_d_path = os.path.join(vi_deformed_dir, f"{stem}.png")
        cv2.imwrite(ir_d_path, ir_d)
        cv2.imwrite(vi_d_path, vi_d)

        ir_flow_path = os.path.join(ir_flow_dir, f"{stem}.npy")
        vi_flow_path = os.path.join(vi_flow_dir, f"{stem}.npy")
        np.save(ir_flow_path, flow_ir)
        np.save(vi_flow_path, flow_vi)

    print("All done!")
    print(f"Deformed IR: {ir_deformed_dir}")
    print(f"Deformed VI: {vi_deformed_dir}")
    print(f"IR flows: {ir_flow_dir}")
    print(f"VI flows: {vi_flow_dir}")

if __name__ == '__main__':
    main()