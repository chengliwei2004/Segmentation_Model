import os
import shutil
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

# -------------------------------------------------------#
#   配置数据集路径
# -------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'
img_dir = os.path.join(VOCdevkit_path, r'/home/yons/clw/deeplabv3-plus-pytorch/VOCdevkit/VOC2007/JPEGImages')
mask_dir = os.path.join(VOCdevkit_path, r'/home/yons/clw/deeplabv3-plus-pytorch/VOCdevkit/VOC2007/SegmentationClass')

brightness_ranges = {
    "_dark": (0.5, 0.85),
    "_bright": (1.15, 1.5)
}

def augment_dataset_randomly():
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg') and not any(suffix in f for suffix in brightness_ranges.keys())]
    
    print(f"总计检测到 {len(img_names)} 张原始图片，开始进行随机区间亮度增强...")
    
    for img_name in tqdm(img_names, desc="Augmenting"):
        base_name = img_name.split('.jpg')[0]
        mask_name = base_name + '.png'
        
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"\n警告: 找不到 {img_name} 对应的掩码文件 {mask_name}，已跳过。")
            continue
            
        try:
            image = Image.open(img_path)
            enhancer = ImageEnhance.Brightness(image)
        except Exception as e:
            print(f"\n读取图片 {img_name} 失败: {e}")
            continue

        # 生成不同亮度的版本
        for suffix, (min_factor, max_factor) in brightness_ranges.items():
            random_factor = random.uniform(min_factor, max_factor)
            
            new_img_name = base_name + suffix + '.jpg'
            new_mask_name = base_name + suffix + '.png'
            
            new_img_path = os.path.join(img_dir, new_img_name)
            new_mask_path = os.path.join(mask_dir, new_mask_name)
            new_image = enhancer.enhance(random_factor)
            new_image.save(new_img_path)
            shutil.copyfile(mask_path, new_mask_path)

    print("\n数据增强完成！")

if __name__ == "__main__":
    augment_dataset_randomly()
