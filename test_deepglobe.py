import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.deepglobe_dataset import DeepGlobeDataset
from model.unet import UNet
from utils.sliding_window import sliding_window_predict_dataset


def optimized_thin(binary_img):
    """使用OpenCV优化的细化算法"""
    try:
        thinned = cv2.ximgproc.thinning(
            binary_img.astype(np.uint8), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return thinned
    except AttributeError:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thinned = binary_img.copy()
        for _ in range(5):
            thinned = cv2.erode(thinned, kernel)
        return thinned


def fast_thin_image(image_path):
    """优化后的标签图像处理流程"""
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")

    _, binary = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    thinned = optimized_thin(dilated)
    return thinned


def find_mask_file(base_name, mask_folder):
    """尝试多种方式找到对应的 mask 文件"""
    mask_candidates = [
        f"{base_name}_mask.png",
        f"{base_name}_sat_mask.png",
    ]

    if base_name.endswith("_sat"):
        mask_candidates.append(f"{base_name[:-4]}_mask.png")

    for mask_filename in mask_candidates:
        candidate_path = os.path.join(mask_folder, mask_filename)
        if os.path.exists(candidate_path):
            return candidate_path

    return None


def batch_calculate_connectivity(pred_folder, mask_folder, log_path="connectivity.log"):
    """批量处理优化版本"""
    start_time = time.time()
    ratio_list = np.array([], dtype=np.float32)
    valid_pairs = 0

    pred_files = (f for f in os.listdir(pred_folder) if f.endswith("_pred.png"))

    with (
        open(log_path, "w") as log_file,
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        for pred_file in pred_files:
            base_name = pred_file[:-9]

            mask_path = find_mask_file(base_name, mask_folder)
            if mask_path is None:
                continue

            try:
                pred_path = os.path.join(pred_folder, pred_file)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                if pred is None:
                    continue

                pred_bin = cv2.threshold(pred, 128, 255, cv2.THRESH_BINARY)[1]
                thinned_gt = fast_thin_image(mask_path)

                if pred_bin.shape != thinned_gt.shape:
                    pred_bin = cv2.resize(
                        pred_bin,
                        (thinned_gt.shape[1], thinned_gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                valid_gt = np.count_nonzero(thinned_gt)
                if valid_gt == 0:
                    continue

                matched = cv2.bitwise_and(thinned_gt, pred_bin)
                ratio = np.count_nonzero(matched) / valid_gt

                ratio_list = np.append(ratio_list, ratio)
                valid_pairs += 1

                log_line = f"{base_name}: {ratio:.4f}\n"
                log_file.write(log_line)

            except Exception as e:
                print(f"Error processing {pred_file}: {str(e)}")
                continue

        avg_ratio = np.mean(ratio_list) if ratio_list.size > 0 else 0.0
        summary = (
            f"\nProcessing time: {time.time() - start_time:.2f}s\n"
            f"Valid pairs: {valid_pairs}\n"
            f"Average connectivity: {avg_ratio:.4f}\n"
        )
        log_file.write(summary)
        print(summary)

    return avg_ratio


def calculate_metrics_with_threshold(pred_folder, mask_folder, threshold=0.5):
    """计算基本的分割指标"""
    tp_total = fp_total = fn_total = tn_total = 0
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith("_pred.png")]

    matched_count = 0
    for filename in pred_files:
        base_name = filename.replace("_pred.png", "")

        mask_path = find_mask_file(base_name, mask_folder)
        if mask_path is None:
            continue

        try:
            mask = np.array(Image.open(mask_path))
            pred = np.array(Image.open(os.path.join(pred_folder, filename)))

            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            if len(pred.shape) == 3:
                pred = pred[:, :, 0]

            if pred.shape != mask.shape:
                pred = cv2.resize(
                    pred,
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_bin = mask > 128
            pred_bin = pred > threshold * 255

            tp = np.sum(pred_bin & mask_bin)
            fp = np.sum(pred_bin & ~mask_bin)
            fn = np.sum(~pred_bin & mask_bin)
            tn = np.sum(~pred_bin & ~mask_bin)

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
            matched_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    epsilon = 1e-10

    iou_road = tp_total / (tp_total + fp_total + fn_total + epsilon)
    iou_background = tn_total / (tn_total + fp_total + fn_total + epsilon)
    miou = (iou_road + iou_background) / 2
    accuracy = (tp_total + tn_total) / (
        tp_total + tn_total + fp_total + fn_total + epsilon
    )

    precision = tp_total / (tp_total + fp_total + epsilon)
    recall = tp_total / (tp_total + fn_total + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return miou, accuracy, precision, recall, f1_score, iou_road, iou_background


def find_best_threshold(pred_folder, mask_folder, thresholds=np.arange(0.1, 0.9, 0.05)):
    """在验证集上搜索最佳阈值"""
    print(f"🔍 搜索最佳阈值，范围: {thresholds[0]:.2f} ~ {thresholds[-1]:.2f}")

    best_miou = 0
    best_threshold = 0.5
    best_metrics = None

    for threshold in tqdm(thresholds, desc="Searching best threshold"):
        miou, accuracy, precision, recall, f1, road_iou, bg_iou = (
            calculate_metrics_with_threshold(pred_folder, mask_folder, threshold)
        )

        if miou > best_miou:
            best_miou = miou
            best_threshold = threshold
            best_metrics = (miou, accuracy, precision, recall, f1, road_iou, bg_iou)
            print(f"  阈值 {threshold:.2f}: mIoU = {miou:.4f} ← 新最佳!")

    print(f"\n✅ 最佳阈值: {best_threshold:.2f}, 最佳 mIoU: {best_miou:.4f}")
    return best_threshold, best_metrics


def generate_predictions(
    model, test_loader, output_folder, device, save_raw_scores=False
):
    """生成预测结果"""
    model.eval()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if save_raw_scores:
        score_folder = output_folder / "scores"
        score_folder.mkdir(parents=True, exist_ok=True)

    print(f"📸 开始生成预测结果，保存到: {output_folder}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images = batch["image"].to(device)
            image_names = batch["image_name"]

            outputs = model(images)
            outputs = outputs[:, 0:1, :, :]
            scores = torch.sigmoid(outputs)

            for i, img_name in enumerate(image_names):
                # 保存时去掉 _sat 后缀，方便匹配 mask
                base_name = img_name
                if base_name.endswith("_sat"):
                    base_name = base_name[:-4]

                # 保存二值预测结果（默认阈值0.5）
                pred_np = (scores[i, 0].cpu().numpy() > 0.5) * 255
                pred_np = pred_np.astype(np.uint8)
                output_path = output_folder / f"{base_name}_pred.png"
                Image.fromarray(pred_np).save(output_path)

                # 保存原始概率图（可选）
                if save_raw_scores:
                    score_np = (scores[i, 0].cpu().numpy() * 255).astype(np.uint8)
                    score_path = score_folder / f"{base_name}_score.png"
                    Image.fromarray(score_np).save(score_path)

    print(f"✅ 预测完成，共生成 {len(list(output_folder.glob('*.png')))} 张预测图")


def main():
    parser = argparse.ArgumentParser(
        description="DeepGlobe 测试脚本 - 生成预测并计算指标"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe",
        help="数据集根目录",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="saved/deepglobe/model_best.pth",
        help="模型权重路径",
    )
    parser.add_argument("--output_dir", type=str, default="Output", help="输出文件夹")
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"], help="测试集分割"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="二值化阈值 (默认: 0.5)"
    )
    parser.add_argument(
        "--find_best_threshold",
        action="store_true",
        help="在验证集上搜索最佳阈值（需要 --split val）",
    )
    parser.add_argument("--save_raw_scores", action="store_true", help="保存原始概率图")
    parser.add_argument(
        "--use_sliding_window",
        action="store_true",
        help="使用滑动窗口预测（适用于大图像）",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=256,
        help="滑动窗口大小（默认: 256，与训练一致）",
    )
    parser.add_argument(
        "--stride", type=int, default=128, help="滑动窗口步长（默认: 128，50%%重叠）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DeepGlobe 道路分割测试")
    print("=" * 60)
    print(f"数据集: {args.data_root}")
    print(f"权重文件: {args.weights}")
    print(f"输出目录: {args.output_dir}")
    print(f"测试集: {args.split}")
    print(f"设备: {args.device}")
    print(f"阈值: {args.threshold}")
    if args.find_best_threshold:
        print(f"搜索最佳阈值: 是")
    if args.use_sliding_window:
        print(f"滑动窗口: 启用 (窗口={args.window_size}, 步长={args.stride})")
    else:
        print(f"滑动窗口: 禁用（使用原始方式）")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_folder = output_dir / "predictions"
    pred_folder.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("\n📦 加载模型...")
    model = UNet(block="BasicBlock").to(args.device)

    checkpoint = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"✅ 模型加载成功 (epoch {checkpoint['epoch']}, mIoU {checkpoint.get('miou', 0):.4f})"
    )

    # 创建测试数据集
    print(f"\n📂 加载 {args.split} 数据集...")
    test_dataset = DeepGlobeDataset(
        datasets_root=args.data_root, split=args.split, data_aug_prob=0.0
    )
    print(f"✅ 测试集样本数: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # 生成预测
    if args.use_sliding_window:
        sliding_window_predict_dataset(
            model=model,
            dataset=test_dataset,
            output_folder=pred_folder,
            device=args.device,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            save_raw_scores=args.save_raw_scores,
        )
    else:
        generate_predictions(
            model, test_loader, pred_folder, args.device, args.save_raw_scores
        )

    # 计算指标
    print("\n" + "=" * 60)
    print("📊 计算评估指标...")
    print("=" * 60)

    mask_folder = Path(args.data_root) / args.split / "seg"

    if args.find_best_threshold and args.split == "val":
        best_threshold, best_metrics = find_best_threshold(
            str(pred_folder), str(mask_folder)
        )
        miou, accuracy, precision, recall, f1, road_iou, bg_iou = best_metrics
        used_threshold = best_threshold
    else:
        miou, accuracy, precision, recall, f1, road_iou, bg_iou = (
            calculate_metrics_with_threshold(
                str(pred_folder), str(mask_folder), args.threshold
            )
        )
        used_threshold = args.threshold

    connectivity = batch_calculate_connectivity(
        str(pred_folder),
        str(mask_folder),
        log_path=str(output_dir / "connectivity.log"),
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("🎉 评估指标计算完成！")
    print("=" * 60)
    print(f"📌 使用阈值: {used_threshold:.2f}")
    print(f"🎯 Precision (精确率): {precision:.4f}")
    print(f"🎯 Accuracy (准确率):  {accuracy:.4f}")
    print(f"🎯 Recall (召回率):    {recall:.4f}")
    print(f"🎯 F1 Score:           {f1:.4f}")
    print(f"🎯 mIoU:               {miou:.4f}")
    print(f"🎯 Road IoU:           {road_iou:.4f}")
    print(f"🎯 Background IoU:     {bg_iou:.4f}")
    print(f"🎯 Connectivity (连通率): {connectivity:.4f}")
    print("=" * 60)

    # 保存结果
    result_file = output_dir / "evaluation_metrics.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("DeepGlobe 道路分割评估指标\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型权重: {args.weights}\n")
        f.write(f"测试集: {args.split}\n")
        f.write(f"预测文件夹: {pred_folder}\n")
        f.write(f"标签文件夹: {mask_folder}\n")
        f.write(f"使用阈值: {used_threshold:.2f}\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("基本指标:\n")
        f.write(f"  Precision (精确率): {precision:.4f}\n")
        f.write(f"  Accuracy (准确率):  {accuracy:.4f}\n")
        f.write(f"  Recall (召回率):    {recall:.4f}\n")
        f.write(f"  F1 Score:           {f1:.4f}\n\n")

        f.write("IoU指标:\n")
        f.write(f"  mIoU:               {miou:.4f}\n")
        f.write(f"  Road IoU:           {road_iou:.4f}\n")
        f.write(f"  Background IoU:     {bg_iou:.4f}\n\n")

        f.write("连通性指标:\n")
        f.write(f"  Connectivity:       {connectivity:.4f}\n")

    print(f"\n📄 结果已保存到: {result_file}")
    print(f"📂 预测结果保存在: {pred_folder}")


if __name__ == "__main__":
    main()
