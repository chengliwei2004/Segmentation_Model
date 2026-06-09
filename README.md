# 茶壶分割模型设计
## 一、项目简介
本项目主要针对高曝光、玻璃反光、低对比度环境等复杂场景下紫砂壶分割问题。本项目的主要流程如下：

+ 数据集的准备
+ 茶壶分割模型构建（基于DeepLabV3算法）
+ 模型优化（加入ECA注意力机制）
+ 模型训练

## 二、数据准备
1. 茶壶数据集：[https://huggingface.co/datasets/AGI-FBHC/ChaHu](https://huggingface.co/datasets/AGI-FBHC/ChaHu)
2. 数据集特点：
+ 图片是高分辨率图片
+ 图片均无小目标
+ 大部分图片都在灯光较亮的场景下，也有部分在较暗环境下
+ 有很多图片中茶壶的颜色与背景颜色类似
+ 因为茶壶在玻璃罩中，所以将近一半的图片中都有出现玻璃反光的问题
3. 数据分类（两类）

```plain
{
  "0": "background",
  "1": "chahu",
}
```

4. 数据存放格式

```plain
项目根目录/
└── VOCdevkit/
    └── VOC2007/
        ├── ImageSets/
        │   └── Segmentation/
        │       ├── train.txt       <-- 训练集图片名称列表 (运行 voc_annotation.py 自动生成)
        │       ├── val.txt         <-- 验证集图片名称列表 (运行 voc_annotation.py 自动生成)
        │       └── test.txt        <-- 测试集图片名称列表 (选配)
        ├── JPEGImages/
        │   ├── JN000001.jpg        <-- 所有的原图全部放在这里
        │   └── JN000002.jpg
        └── SegmentationClass/
            ├── JN000001.png        <-- 所有的掩码标签图全部放在这里，且必须是单通道PNG
            └── JN000002.png
```

5. 数据清洗与构建

```plain
"""
该代码主要针对开源数据标注中JPG掩码有损压缩噪点问题进行二值化阈值过滤（jpg-->png）
"""
python Data_classification_and_transformation.py # 完成数据清洗
python voc_annotation.py # 完成训练集和验证集的划分
```

6. 数据增强（主要针对不同亮度环境进行增强）

```plain
"""
通过使用随机亮度对每一张图片完成变亮和变暗操作
"""
python data_augmentation.py
```

## 三、 模型优化
1. 基础模型架构

 在网络架构选型上，本项目基于主流的DeepLabV3+分割框架进行构建，采用mobilenet作为主干网络。

2. 模型优化
+ **为什么选择ECA注意力机制：**ECA注意力机制采用无降维的一维卷积来实现跨通道交互，而且对于模型来说参数量仅增加几十个 ，就能够自适应地精准抑制玻璃高光反光、增强暗部真实边缘，同时最大程度地保留紫砂壶微妙的泥料纹理细节。  
+ **在哪里加入ECA注意力制：**ECA 注意力机制被插入浅层特征与 ASPP 深层特征完成物理拼接之后，且在进入最终的融合卷积之前。（详细见下面DeepLabV3+框架图）
+ **在该处加入ECA注意力机制的优势：** 在解码器浅深层特征拼接后加入ECA机制能够在局部细节与全局语义真正融合前进行动态过滤反光噪点。ECA注意力机制利用该位置的全知视角，通过无降维的自适应一维卷积进行跨通道交互，精准压制对玻璃高光过度兴奋的通道，同时还能够放大刻画真实暗部轮廓的通道。该操作能够最大程度地保留紫砂壶极其微妙的泥料纹理细节，从而显著提升复杂光影场景下的分割鲁棒性。  
3. 下图为原始DeepLabV3+框架图和加入ECA注意力机制之后的DeepLabV3+框架图

<img src="https://cdn.nlark.com/yuque/0/2026/png/40833697/1781008295698-314bf770-e81e-4d18-9462-f309a050ed00.png" width="1536" title="" crop="0,0,1,1" id="u15a5cd71" class="ne-image">

## 四、 模型训练
1. 环境安装

```plain
pip install -r requirements.txt
```

2. 训练参数配置

```plain
--- 核心任务配置 ---
num_classes     = 2          # 类别数 + 1 (背景=0，目标=1)
backbone        = "mobilenet" # 特征提取主干网络
pretrained      = True       # 启用预训练权重加载

# --- 训练超参数 ---
Init_Epoch          = 0
Freeze_Epoch        = 50     # 冻结主干网络训练的 Epoch 数
UnFreeze_Epoch      = 150    # 总训练 Epoch 数
Freeze_batch_size   = 32     # 冻结阶段 Batch Size
Unfreeze_batch_size = 16     # 解冻阶段 Batch Size
Init_lr             = 0.007  # 初始学习率 (使用 SGD 优化器)
```

3. 模型训练

```plain
nohup python train.py > deeplab_ECA.log 2>&1 &  # 后台训练
tail -f deeplab_ECA.log # 训练日志查看
```

4. 模型输出

```plain
logs/best_epoch_weights.pth - 最佳验证集损失模型权重
logs/last_epoch_weights.pth - 最终轮次训练模型权重
logs/epoch_loss.txt - 训练集损失历史记录文本
logs/epoch_val_loss.txt - 验证集损失历史记录文本
logs/events.out.tfevents.* - TensorBoard 训练与验证曲线数据日志
deeplab_ECA.log - 后台运行的完整控制台训练日志文件
```

## 五、 实验结果
1. 训练效果图
+ Loss 曲线图（未加入ECA注意力机制）

<img src="https://cdn.nlark.com/yuque/0/2026/png/40833697/1781012842636-c644039c-1fb2-4a5f-8bd0-f427d6a40a86.png" width="715" title="" crop="0,0,1,1" id="u05a80878" class="ne-image">

+ Loss 曲线图（加入ECA注意力机制）

<img src="https://cdn.nlark.com/yuque/0/2026/png/40833697/1781012923176-2793cc23-8287-49ef-afd0-578451f71289.png" width="715" title="" crop="0,0,1,1" id="uc2756f29" class="ne-image">

2. 总结 

从两幅Loss曲线对比图可以直观地看出，加入ECA注意力机制后，模型在训练过程中的核心优势在于极大地提升了验证集表现的稳定性和模型的整体泛化能力。在未加入ECA的基础网络中，验证集损失曲线出现了多次剧烈的上下震荡与尖峰，这表明模型在面对未知数据时容易受到批次噪声的干扰，且极易在训练后期陷入局部过拟合状态；而引入ECA模块后，验证集损失曲线的走势变得显著平滑，先前的剧烈波动被有效压制，损失值能够持续稳定在一个较低的收敛区间内。两幅曲线图证明了ECA机制能够通过自适应地提取关键通道特征并抑制无关背景噪声。ECA注意力机制的加入不仅能够模型滤除了训练过程中的冗余干扰信息，起到了良好的正则化效果，还可以使得整个训练收敛过程更加稳健最终赋予了模型更强的抗过拟合能力与应对新数据的可靠性。  

2. 测试参数列表

| **<font style="color:rgb(31, 31, 31);">模型配置</font>** | **<font style="color:rgb(31, 31, 31);">Max mIoU (最高平均交并比)</font>** | **<font style="color:rgb(31, 31, 31);">Final mIoU (最终轮次)</font>** | **<font style="color:rgb(31, 31, 31);">最后10轮平均 mIoU</font>** | **<font style="color:rgb(31, 31, 31);">Min Val Loss (最佳验证集损失)</font>** | **<font style="color:rgb(31, 31, 31);">Final Val Loss (最终轮次验证集损失)</font>** |
| --- | --- | --- | --- | --- | --- |
| **<font style="color:rgb(31, 31, 31);">Baseline</font>**<font style="color:rgb(31, 31, 31);"> (原生 DeepLab)</font> | <font style="color:rgb(31, 31, 31);">96.7766%</font> | <font style="color:rgb(31, 31, 31);">96.4892%</font> | <font style="color:rgb(31, 31, 31);">96.5133%</font> | **<font style="color:rgb(31, 31, 31);">0.0112</font>** | <font style="color:rgb(31, 31, 31);">0.0225</font> |
| **<font style="color:rgb(31, 31, 31);">Baseline + ECA</font>** | **<font style="color:rgb(31, 31, 31);">96.8390%</font>** | **<font style="color:rgb(31, 31, 31);">96.5187%</font>** | **<font style="color:rgb(31, 31, 31);">96.5257%</font>** | <font style="color:rgb(31, 31, 31);">0.0114</font> | **<font style="color:rgb(31, 31, 31);">0.0182</font>** |


3. 推理结果图

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013545286-0296b72f-b0b3-4c39-940e-7d40a150f052.jpeg" width="309" title="" crop="0,0,1,1" id="u3e885ffe" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013784908-26efb1f5-afe2-4b5e-b952-ec59feacdd3d.jpeg" width="309" title="" crop="0,0,1,1" id="u85ab66ba" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013543376-83f00173-f54c-4984-b380-46602f0e57d4.jpeg" width="309" title="" crop="0,0,1,1" id="ueeab7a27" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013829561-2f1c556f-b41b-45b4-94b7-4ba183183a2b.jpeg" width="309" title="" crop="0,0,1,1" id="ufca8af27" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013543295-64855e03-203a-4146-9247-39d6d713d339.jpeg" width="309" title="" crop="0,0,1,1" id="ua266aa86" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013844051-6c9e42d2-4090-406d-9420-5e94537ee991.jpeg" width="309" title="" crop="0,0,1,1" id="u1fd39937" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013543220-901be49c-7df8-4213-9028-4bd790b16190.jpeg" width="309" title="" crop="0,0,1,1" id="ubfcb86e9" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013854605-3278e5d6-6665-45e7-b2e2-e61328fbe910.jpeg" width="309" title="" crop="0,0,1,1" id="u5777a936" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013543318-00b8243f-da36-48ff-80a3-e61e728657d8.jpeg" width="309" title="" crop="0,0,1,1" id="u91763b33" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013864714-d219914e-5d49-490a-8eb0-026fb0323eb9.jpeg" width="309" title="" crop="0,0,1,1" id="u2d03c730" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013544373-30636eb8-4732-419f-9fb9-bf1ba94741ab.jpeg" width="309" title="" crop="0,0,1,1" id="uf7b2c93e" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013877117-09495ae4-f596-4cd0-84ed-cd4852ff5121.jpeg" width="309" title="" crop="0,0,1,1" id="u407e847b" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013545290-376d759c-64a1-4ab1-b005-523a02c95f75.jpeg" width="309" title="" crop="0,0,1,1" id="u9ecfbf35" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013893807-59dcebd2-8966-44cc-af7c-b4f02bdbc4d0.jpeg" width="309" title="" crop="0,0,1,1" id="u56d61cb5" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013545786-97c3313b-7c44-43ee-85e4-f30b867cfbcc.jpeg" width="309" title="" crop="0,0,1,1" id="u2c69a636" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013906277-a02c6461-4dae-4bc9-83f7-b441e631c3e8.jpeg" width="309" title="" crop="0,0,1,1" id="ubf06808f" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013548355-1d6d4a35-31be-49be-afc0-681d2430cbda.jpeg" width="309" title="" crop="0,0,1,1" id="u9f90884b" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013926422-e1ccc11f-50f3-4160-b5a7-9b8897b25ed1.jpeg" width="309" title="" crop="0,0,1,1" id="ubc780e65" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013547992-098e78e0-f27e-473d-b9fc-5c3114832c7c.jpeg" width="309" title="" crop="0,0,1,1" id="ubcaa8463" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/jpeg/40833697/1781013944652-27a01d95-af23-49e6-beb5-476ea7dbad16.jpeg" width="309" title="" crop="0,0,1,1" id="u0887a15b" class="ne-image">

## 六、 核心代码
1. 数据增强

```python
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
```

2. ECA注意力机制的实例化

```python
self.shortcut_conv = nn.Sequential(
    nn.Conv2d(low_level_channels, 48, 1),
    nn.BatchNorm2d(48),
    nn.ReLU(inplace=True)
)       
self.eca = ECALayer(channel=304)

self.cat_conv = nn.Sequential(
    nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Conv2d(256, 256, 3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.1),
)
self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
```

3.  前向传播（Forward）中的特征拦截  

```python
def forward(self, x):
    H, W = x.size(2), x.size(3)
    
    # 1. 提取浅层特征与深层特征
    low_level_features, x = self.backbone(x)
    x = self.aspp(x)
    
    # 2. 特征对齐（降维与双线性插值上采样）
    low_level_features = self.shortcut_conv(low_level_features)
    x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
    
    # 3. 特征物理拼接 [Batch_size, 304, H/4, W/4]
    cat_features = torch.cat((x, low_level_features), dim=1)
    
    # 通过 ECA 注意力机制进行跨通道特征重标定
    attention_features = self.eca(cat_features)
    
    # 4. 融合加权后的特征并输出预测结果
    x = self.cat_conv(attention_features)
    x = self.cls_conv(x)
    
    # 5. 还原至原图分辨率
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
```
