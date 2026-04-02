import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms

from base import BaseDataset
import data.datautils as util

crop_size = 256


class DeepGlobeDataset(BaseDataset):
    """
    DeepGlobe 道路分割数据集
    数据集结构:
        deepglobe/
        ├── train/
        │   ├── data/ (图像)
        │   └── seg/ (标签)
        ├── val/
        │   ├── data/
        │   └── seg/
        └── test/
            ├── data/
            └── seg/
    """

    def __init__(
        self,
        datasets_root,
        split="train",
        data_aug_prob=0.5,
        mean="[0.485, 0.456, 0.406]",
        std="[0.229, 0.224, 0.225]",
        seed=1234,
    ):
        """
        Args:
            datasets_root: 数据集根目录
            split: 'train', 'val', 或 'test'
            data_aug_prob: 数据增强概率
            mean: 归一化均值
            std: 归一化标准差
            seed: 随机种子
        """
        super(DeepGlobeDataset, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.split = split

        # 直接扫描目录，不使用 txt 文件
        split_dir = self.datasets_root / split
        img_dir = split_dir / "data"
        mask_dir = split_dir / "seg"

        self.image_paths = []
        self.mask_paths = []

        if img_dir.exists() and mask_dir.exists():
            image_files = sorted(list(img_dir.glob("*.jpg")))

            for img_file in image_files:
                # 从图像文件名生成标签文件名：xxx_sat.jpg -> xxx_mask.png
                img_name = img_file.stem  # xxx_sat
                if img_name.endswith("_sat"):
                    mask_name = img_name[:-4] + "_mask.png"  # xxx_mask.png
                else:
                    mask_name = img_name + "_mask.png"

                mask_file = mask_dir / mask_name

                if mask_file.exists():
                    self.image_paths.append(str(img_file))
                    self.mask_paths.append(str(mask_file))

        print(f"Loaded {len(self.image_paths)} samples for {split}")

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.split == "train" and self.data_aug_prob > 0:
            self.transform = transforms.Compose(
                [
                    util.RandomCrop(crop_size),
                    util.Jitter_HSV(self.data_aug_prob),
                    util.RandomHorizontalFlip(self.data_aug_prob),
                    util.RandomVerticleFlip(self.data_aug_prob),
                    util.RandomRotate90(self.data_aug_prob),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 加载图像
        image_src = self.image_paths[idx]
        image_name = Path(image_src).stem
        image = cv2.imread(image_src, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"无法读取图像: {image_src}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载标签
        mask_src = self.mask_paths[idx]
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"无法读取标签: {mask_src}")

        sample = {"image_name": image_name, "image": image, "mask": mask}

        # 在线数据增强
        if self.transform:
            sample = self.transform(sample)

        # 二值化
        _, sample["mask"] = cv2.threshold(sample["mask"], 127, 1, cv2.THRESH_BINARY)

        # 转换为 tensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample["image"] = self.normalize(sample["image"])

        return sample
