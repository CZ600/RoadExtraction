import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from data.deepglobe_dataset import DeepGlobeDataset
from model.unet import UNet
from model.loss import dice_bce_loss
from model.metrics import IoU


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="DeepGlobe 道路分割训练脚本")
    parser.add_argument(
        "--data_root",
        type=str,
        default="D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe",
        help="数据集根目录",
    )
    parser.add_argument(
        "--save_dir", type=str, default="saved/deepglobe", help="模型保存目录"
    )
    parser.add_argument(
        "--log_dir", type=str, default="saved/logs", help="日志保存目录"
    )
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--val_batch_size", type=int, default=4, help="验证批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--save_period", type=int, default=5, help="保存间隔（轮数）")
    parser.add_argument("--input_size", type=int, default=256, help="输入图像大小")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs[:, 0:1, :, :]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        iou = IoU(preds, masks)

        total_loss += loss.item()
        total_iou += float(iou)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "IoU": f"{float(iou):.4f}"})

        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/loss", loss.item(), global_step)
        writer.add_scalar("Train/IoU", float(iou), global_step)

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_intersection = 0.0
    total_union = 0.0
    smooth = 1e-6

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            outputs = outputs[:, 0:1, :, :]
            loss = criterion(outputs, masks)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            iou = IoU(preds, masks)

            total_loss += loss.item()
            total_iou += float(iou)

            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item() - intersection
            total_intersection += intersection
            total_union += union

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "IoU": f"{float(iou):.4f}"})

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    miou = (total_intersection + smooth) / (total_union + smooth)

    writer.add_scalar("Val/loss", avg_loss, epoch)
    writer.add_scalar("Val/IoU", avg_iou, epoch)
    writer.add_scalar("Val/mIoU", miou, epoch)

    return avg_loss, avg_iou, miou


def save_checkpoint(model, optimizer, epoch, miou, save_path, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "miou": miou,
    }
    torch.save(checkpoint, save_path)
    if is_best:
        best_path = Path(save_path).parent / "model_best.pth"
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型到: {best_path}")


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepGlobe 道路分割训练")
    print("=" * 60)
    print(f"数据集根目录: {args.data_root}")
    print(f"模型保存目录: {save_dir}")
    print(f"训练设备: {args.device}")
    print(f"训练轮数: {args.epochs}")
    print(f"训练批次大小: {args.batch_size}")
    print(f"验证批次大小: {args.val_batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 60)

    writer = SummaryWriter(log_dir=str(log_dir))

    print("\n正在创建数据集...")
    train_dataset = DeepGlobeDataset(
        datasets_root=args.data_root, split="train", data_aug_prob=0.5
    )

    val_dataset = DeepGlobeDataset(
        datasets_root=args.data_root, split="val", data_aug_prob=0.0
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\n正在创建模型...")
    model = UNet(block="BasicBlock").to(args.device)

    criterion = dice_bce_loss

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        print(f"\n正在从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("miou", 0.0)
        print(f"从 epoch {start_epoch} 开始恢复，最佳 mIoU: {best_miou:.4f}")

    print("\n开始训练...")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, writer
        )

        val_loss, val_iou, val_miou = validate(
            model, val_loader, criterion, args.device, epoch, writer
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/lr", current_lr, epoch)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(
            f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, mIoU: {val_miou:.4f}"
        )
        print(f"  LR: {current_lr:.6f}")

        if (epoch + 1) % args.save_period == 0:
            save_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(model, optimizer, epoch, val_miou, save_path)
            print(f"保存检查点到: {save_path}")

        if val_miou > best_miou:
            best_miou = val_miou
            save_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(model, optimizer, epoch, val_miou, save_path, is_best=True)
            print(f"新的最佳 mIoU: {best_miou:.4f}")

        print("-" * 60)

    print("\n训练完成!")
    print(f"最佳验证 mIoU: {best_miou:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
