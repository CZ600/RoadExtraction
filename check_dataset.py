from pathlib import Path


def check_dataset_structure():
    print("=" * 60)
    print("检查 DeepGlobe 数据集结构")
    print("=" * 60)

    data_root = Path("D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe")

    if not data_root.exists():
        print(f"错误: 数据集路径不存在: {data_root}")
        return False

    print(f"\n数据集根目录: {data_root}")

    # 检查列表文件
    print("\n检查列表文件:")
    for split in ["train", "val", "test"]:
        list_file = data_root / f"{split}.txt"
        if list_file.exists():
            with open(list_file, "r") as f:
                lines = f.readlines()
                print(f"  {split}.txt: {len(lines)} 行")
        else:
            print(f"  {split}.txt: 不存在")

    # 检查目录结构
    print("\n检查目录结构:")
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if split_dir.exists():
            print(f"\n  {split}/")

            # 检查 data 目录
            data_dir = split_dir / "data"
            if data_dir.exists():
                images = list(data_dir.glob("*.jpg"))
                print(f"    data/: {len(images)} 张图像")
            else:
                print(f"    data/: 不存在")

            # 检查 seg 目录
            seg_dir = split_dir / "seg"
            if seg_dir.exists():
                masks = list(seg_dir.glob("*.png"))
                print(f"    seg/: {len(masks)} 个标签")
            else:
                print(f"    seg/: 不存在")
        else:
            print(f"\n  {split}/: 不存在")

    print("\n" + "=" * 60)
    print("结构检查完成!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    check_dataset_structure()
