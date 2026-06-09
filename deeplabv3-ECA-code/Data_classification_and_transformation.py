import os
import shutil
import numpy as np
from PIL import Image


def organize_and_fix_dataset(source_dir, output_dir, threshold=127, target_value=1):
    """
    整理混放的数据集，并将JPG格式的掩码图修复转换为符合DeepLabV3+要求的单通道PNG标签图。

    :param source_dir: 包含原始 .jpg 和 .mask.jpg 的源文件夹路径
    :param output_dir: 整理后的数据集输出根路径
    :param threshold: 二值化阈值，用于消除JPG有损压缩带来的边缘噪点
    :param target_value: 茶壶目标在标签图中的像素值（单目标分割通常设为类别索引 1）
    """
    # 创建 DeepLab 标准子目录
    images_out_dir = os.path.join(output_dir, 'images')
    masks_out_dir = os.path.join(output_dir, 'labels')

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)

    # 获取所有文件并筛选出所有的掩码文件
    all_files = os.listdir(source_dir)
    mask_files = [f for f in all_files if f.endswith('.mask.jpg')]

    print(f"找到 {len(mask_files)} 组图像对，开始处理...")

    success_count = 0
    for mask_file in mask_files:
        # 获取基础文件名，例如 JN000001
        base_name = mask_file.replace('.mask.jpg', '')
        img_file = base_name + '.jpg'

        mask_path = os.path.join(source_dir, mask_file)
        img_path = os.path.join(source_dir, img_file)

        # 检查原图是否存在，防止数据集缺失错位
        if not os.path.exists(img_path):
            print(f" 警告: 找不到对应的原图 {img_file}，已跳过。")
            continue

        # 1. 复制原图到 images 文件夹
        shutil.copy(img_path, os.path.join(images_out_dir, img_file))

        # 2. 处理 JPG 掩码图并转换为无损 PNG
        # 使用 PIL 打开并转换为单通道灰度图 ('L' 模式)
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)

        # 核心去噪：通过阈值分割过滤掉 JPG 压缩在边缘产生的过渡噪点
        # 大于阈值的判定为茶壶(赋予 target_value)，小于等于的判定为背景(0)
        binary_mask = np.where(mask_array > threshold, target_value, 0).astype(np.uint8)

        # 转换为 PIL 图像并保存为无损 PNG
        # 此时文件名去掉了 .mask 变成了 JN000001.png，完美对应原图
        output_mask_img = Image.fromarray(binary_mask)
        output_mask_path = os.path.join(masks_out_dir, base_name + '.png')
        output_mask_img.save(output_mask_path)

        success_count += 1

    print("\n" + "=" * 30)
    print(f"数据整理完成！共成功转换 {success_count} 组数据。")
    print(f" 训练原图目录: {images_out_dir} (格式: .jpg)")
    print(f" 训练标签目录: {masks_out_dir} (格式: .png, 单通道类别索引)")
    print("=" * 30)


if __name__ == '__main__':
    # ================= 配置区域 =================
    # 请将这里的路径修改为你电脑上的实际文件夹路径
    SOURCE_DIR = "./image"  # 你当前存放这堆混合图片的文件夹
    OUTPUT_DIR = "./deeplab_dataset"  # 你希望输出规范化数据集的目标文件夹

    # 执行整理程序
    organize_and_fix_dataset(SOURCE_DIR, OUTPUT_DIR, threshold=127, target_value=1)