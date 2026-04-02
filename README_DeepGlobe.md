# DeepGlobe 道路分割训练使用说明

## 文件说明

| 文件 | 说明 |
|------|------|
| `data/deepglobe_dataset.py` | DeepGlobe 数据集加载类 |
| `train_deepglobe.py` | 易于使用的训练脚本 |
| `config_deepglobe.json` | 配置文件 |
| `test_deepglobe_dataset.py` | 数据集测试脚本 |

## 快速开始

### 1. 测试数据集加载

首先运行测试脚本验证数据集是否能正常加载：

```bash
python test_deepglobe_dataset.py
```

### 2. 开始训练

使用默认参数训练：

```bash
python train_deepglobe.py
```

### 3. 自定义参数训练

```bash
python train_deepglobe.py \
    --data_root "D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe" \
    --epochs 80 \
    --batch_size 8 \
    --lr 0.001 \
    --save_period 5
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | 数据集路径 | DeepGlobe 数据集根目录 |
| `--save_dir` | saved/deepglobe | 模型保存目录 |
| `--log_dir` | saved/logs | TensorBoard 日志目录 |
| `--epochs` | 80 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--save_period` | 5 | 检查点保存间隔（轮数） |
| `--input_size` | 256 | 输入图像大小 |
| `--num_workers` | 4 | 数据加载线程数 |
| `--resume` | None | 恢复训练的检查点路径 |
| `--device` | auto | 训练设备（cuda/cpu） |

## 训练特性

- ✅ **80 轮训练**：完整的训练周期
- ✅ **每 5 轮保存检查点**：定期保存模型
- ✅ **保存最佳模型**：自动保存验证集 mIoU 最高的模型
- ✅ **TensorBoard 日志**：实时监控训练过程
- ✅ **数据增强**：训练时使用随机裁剪、翻转、旋转等
- ✅ **学习率调度**：余弦退火学习率调度
- ✅ **混合损失函数**：Dice + BCE 组合损失

## 查看训练过程

启动 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir saved/logs
```

然后在浏览器打开：`http://localhost:6006`

## 输出文件

训练完成后，会在 `saved/deepglobe/` 目录下生成：

- `checkpoint_epoch_5.pth` - 第 5 轮检查点
- `checkpoint_epoch_10.pth` - 第 10 轮检查点
- ...
- `model_best.pth` - 验证集 mIoU 最高的最佳模型

## 恢复训练

如果训练中断，可以从检查点恢复：

```bash
python train_deepglobe.py --resume saved/deepglobe/checkpoint_epoch_40.pth
```

## 数据集结构

```
deepglobe/
├── train.txt          # 训练集文件列表
├── val.txt            # 验证集文件列表
├── test.txt           # 测试集文件列表
├── train/
│   ├── data/          # 训练图像 (*.jpg)
│   └── seg/           # 训练标签 (*_mask.png)
├── val/
│   ├── data/
│   └── seg/
└── test/
    ├── data/
    └── seg/
```

## 模型架构

默认使用 **U-Net** 架构，支持：
- BasicBlock（默认）
- SEBasicBlock（带 SE 注意力）
- GABasicBlock（带全局注意力）

## 常见问题

### Q: 显存不足怎么办？
A: 减小 `--batch_size`，例如改为 4 或 2。

### Q: 如何使用其他模型？
A: 修改 `train_deepglobe.py` 中的模型导入和创建部分。

### Q: 训练速度慢？
A: 增加 `--num_workers` 或减小 `--batch_size`。
