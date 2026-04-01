from PIL import Image
import os
import random

def augment_image_pairs(input_ir_folder, input_vis_folder,
                        output_ir_folder, output_vis_folder,
                        num_augmentations=5, crop_size=256, fill_color=0,
                        ir_ext='.png', vis_ext='.png'):
    os.makedirs(output_ir_folder, exist_ok=True)
    os.makedirs(output_vis_folder, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    ir_files = [f for f in os.listdir(input_ir_folder) if f.lower().endswith(valid_exts)]

    for ir_file in ir_files:
        base_name = os.path.splitext(ir_file)[0]
        ir_path = os.path.join(input_ir_folder, ir_file)
        vis_path = os.path.join(input_vis_folder, base_name + vis_ext)

        if not os.path.exists(vis_path):
            print(f"Visible image {vis_path} not found, skipping...")
            continue

        try:
            ir_img = Image.open(ir_path)
            vis_img = Image.open(vis_path)
        except Exception as e:
            print(f"Error opening image pair: {e}")
            continue

        ir_size = ir_img.size
        if vis_img.size != ir_size:
            print(f"Resizing visible image to match infrared size {ir_size}")
            vis_img = vis_img.resize(ir_size)

        for aug_idx in range(num_augmentations):
            ir = ir_img.copy()
            vis = vis_img.copy()

            width, height = ir.size
            new_width = max(width, crop_size)
            new_height = max(height, crop_size)

            pad_w = max(new_width - width, 0)
            pad_h = max(new_height - height, 0)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

            ir_padded = Image.new(ir.mode, (new_width, new_height), fill_color)
            ir_padded.paste(ir, (padding[0], padding[1]))

            vis_padded = Image.new(vis.mode, (new_width, new_height), fill_color)
            vis_padded.paste(vis, (padding[0], padding[1]))

            x = random.randint(0, max(new_width - crop_size, 0))
            y = random.randint(0, max(new_height - crop_size, 0))

            ir_crop = ir_padded.crop((x, y, x + crop_size, y + crop_size))
            vis_crop = vis_padded.crop((x, y, x + crop_size, y + crop_size))

            angle = random.choice([0, 90, 180, 270])
            ir_rot = ir_crop.rotate(angle)
            vis_rot = vis_crop.rotate(angle)

            if random.random() > 0.5:
                ir_rot = ir_rot.transpose(Image.FLIP_LEFT_RIGHT)
                vis_rot = vis_rot.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                ir_rot = ir_rot.transpose(Image.FLIP_TOP_BOTTOM)
                vis_rot = vis_rot.transpose(Image.FLIP_TOP_BOTTOM)

            ir_gray = ir_rot.convert('L')
            vis_color = vis_rot

            ir_output = os.path.join(output_ir_folder, f"{base_name}_aug{aug_idx}{ir_ext}")
            vis_output = os.path.join(output_vis_folder, f"{base_name}_aug{aug_idx}{vis_ext}")

            ir_gray.save(ir_output)
            vis_color.save(vis_output)

if __name__ == "__main__":
    augment_image_pairs(
        input_ir_folder=r'C:\Users\fybzsazzz\Desktop\WaveletFusion\data\RoadScene\ir',
        input_vis_folder=r'C:\Users\fybzsazzz\Desktop\WaveletFusion\data\RoadScene\vi',
        output_ir_folder=r'data/data/ir',
        output_vis_folder=r'data/data/vi',
        num_augmentations=20,
        ir_ext='.jpg',
        vis_ext='.jpg'
    )