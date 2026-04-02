import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class SlidingWindowPredictor:
    """
    滑动窗口预测器
    用于将大图像分割成小块进行预测，然后合并结果
    """

    def __init__(self, model, window_size=256, stride=128, device="cuda", batch_size=4):
        """
        Args:
            model: 训练好的模型
            window_size: 滑动窗口大小（与训练时一致）
            stride: 步长（重叠区域 = window_size - stride）
            device: 设备
            batch_size: 批次大小
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.batch_size = batch_size

    def predict(self, image):
        """
        对单张大图像进行滑动窗口预测

        Args:
            image: numpy数组，形状 (H, W, 3)，RGB格式

        Returns:
            prediction: 预测结果，形状 (H, W)，值范围 [0, 1]
        """
        h, w = image.shape[:2]

        # 计算需要padding的尺寸，确保能被窗口覆盖
        pad_h = (
            max(0, self.window_size - (h % self.stride)) if h % self.stride != 0 else 0
        )
        pad_w = (
            max(0, self.window_size - (w % self.stride)) if w % self.stride != 0 else 0
        )

        # 对图像进行padding
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        h_pad, w_pad = image_pad.shape[:2]

        # 初始化累加器和权重图
        pred_accum = np.zeros((h_pad, w_pad), dtype=np.float32)
        weight_accum = np.zeros((h_pad, w_pad), dtype=np.float32)

        # 生成所有窗口位置
        windows = []
        for y in range(0, h_pad - self.window_size + 1, self.stride):
            for x in range(0, w_pad - self.window_size + 1, self.stride):
                windows.append((y, x))

        # 批量处理窗口
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(
                range(0, len(windows), self.batch_size), desc="Sliding window"
            ):
                batch_windows = windows[i : i + self.batch_size]
                batch_tensors = []

                # 准备批次数据
                for y, x in batch_windows:
                    window = image_pad[
                        y : y + self.window_size, x : x + self.window_size
                    ]
                    # 归一化（与数据集保持一致）
                    window_tensor = torch.from_numpy(
                        (window / 255.0).transpose((2, 0, 1)).astype(np.float32)
                    )
                    # 使用ImageNet均值和标准差归一化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    window_tensor = (window_tensor - mean) / std
                    batch_tensors.append(window_tensor)

                # 堆叠并移动到设备
                batch_tensor = torch.stack(batch_tensors).to(self.device)

                # 模型预测
                outputs = self.model(batch_tensor)
                outputs = outputs[:, 0:1, :, :]  # 取第一个通道
                scores = torch.sigmoid(outputs)

                # 将预测结果放回累加器
                for idx, (y, x) in enumerate(batch_windows):
                    pred_np = scores[idx, 0].cpu().numpy()

                    # 创建权重图（边缘权重较低，中心权重较高）
                    weight = self._create_window_weight(self.window_size)

                    # 累加预测结果和权重
                    pred_accum[y : y + self.window_size, x : x + self.window_size] += (
                        pred_np * weight
                    )
                    weight_accum[
                        y : y + self.window_size, x : x + self.window_size
                    ] += weight

        # 归一化预测结果
        pred_accum = pred_accum / np.maximum(weight_accum, 1e-8)

        # 裁剪回原始尺寸
        prediction = pred_accum[:h, :w]

        return prediction

    def _create_window_weight(self, size):
        """
        创建窗口权重图（高斯分布，中心权重高，边缘权重低）

        Args:
            size: 窗口大小

        Returns:
            weight: 权重图，形状 (size, size)
        """
        sigma = size / 6.0
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        xx, yy = np.meshgrid(x, y)
        weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return weight


def sliding_window_predict_dataset(
    model,
    dataset,
    output_folder,
    device="cuda",
    window_size=256,
    stride=128,
    batch_size=4,
    save_raw_scores=False,
):
    """
    对整个数据集进行滑动窗口预测

    Args:
        model: 训练好的模型
        dataset: 数据集对象
        output_folder: 输出文件夹
        device: 设备
        window_size: 滑动窗口大小
        stride: 步长
        batch_size: 批次大小
        save_raw_scores: 是否保存原始概率图
    """
    from pathlib import Path
    from PIL import Image

    model.eval()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if save_raw_scores:
        score_folder = output_folder / "scores"
        score_folder.mkdir(parents=True, exist_ok=True)

    # 创建滑动窗口预测器
    predictor = SlidingWindowPredictor(
        model=model,
        window_size=window_size,
        stride=stride,
        device=device,
        batch_size=batch_size,
    )

    print(f"📸 开始滑动窗口预测，保存到: {output_folder}")
    print(f"   窗口大小: {window_size}x{window_size}, 步长: {stride}")

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        # 获取原始图像（不使用数据集的transform，直接读取）
        sample = dataset[idx]
        image_name = sample["image_name"]

        # 直接从文件读取原始图像
        image_path = dataset.image_paths[idx]
        import cv2

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 滑动窗口预测
        prediction = predictor.predict(image)

        # 保存时去掉 _sat 后缀
        base_name = image_name
        if base_name.endswith("_sat"):
            base_name = base_name[:-4]

        # 保存二值预测结果（默认阈值0.5）
        pred_np = (prediction > 0.5) * 255
        pred_np = pred_np.astype(np.uint8)
        output_path = output_folder / f"{base_name}_pred.png"
        Image.fromarray(pred_np).save(output_path)

        # 保存原始概率图（可选）
        if save_raw_scores:
            score_np = (prediction * 255).astype(np.uint8)
            score_path = score_folder / f"{base_name}_score.png"
            Image.fromarray(score_np).save(score_path)

    print(f"✅ 预测完成，共生成 {len(list(output_folder.glob('*.png')))} 张预测图")
