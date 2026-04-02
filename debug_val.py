import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

data_root = Path("D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe")

print("=" * 60)
print("调试验证集路径问题")
print("=" * 60)

# 检查 val.txt
val_txt = data_root / "val.txt"
print(f"\nval.txt 存在: {val_txt.exists()}")

if val_txt.exists():
    with open(val_txt, "r") as f:
        lines = f.readlines()
        print(f"val.txt 有 {len(lines)} 行")
        print(f"\n前 5 行:")
        for i, line in enumerate(lines[:5]):
            print(f"  {i + 1}. {line.strip()}")

            # 测试路径
            img_rel, mask_rel = line.strip().split()

            # 原始路径
            img_path1 = data_root / img_rel
            mask_path1 = data_root / mask_rel
            print(f"     原始图像路径: {img_path1} - 存在: {img_path1.exists()}")
            print(f"     原始标签路径: {mask_path1} - 存在: {mask_path1.exists()}")

            # 修正后路径
            img_parts = list(Path(img_rel).parts)
            mask_parts = list(Path(mask_rel).parts)

            img_parts.insert(1, "data")
            mask_parts.insert(1, "seg")

            img_path2 = data_root / Path(*img_parts)
            mask_path2 = data_root / Path(*mask_parts)
            print(f"     修正图像路径: {img_path2} - 存在: {img_path2.exists()}")
            print(f"     修正标签路径: {mask_path2} - 存在: {mask_path2.exists()}")

# 直接检查目录
print(f"\n检查 val/data/:")
val_data_dir = data_root / "val" / "data"
val_seg_dir = data_root / "val" / "seg"

print(f"  val/data/ 存在: {val_data_dir.exists()}")
print(f"  val/seg/ 存在: {val_seg_dir.exists()}")

if val_data_dir.exists():
    jpgs = list(val_data_dir.glob("*.jpg"))
    print(f"  val/data/ 下有 {len(jpgs)} 个 jpg 文件")

if val_seg_dir.exists():
    pngs = list(val_seg_dir.glob("*.png"))
    print(f"  val/seg/ 下有 {len(pngs)} 个 png 文件")
