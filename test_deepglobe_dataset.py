import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.deepglobe_dataset import DeepGlobeDataset


def test_dataset():
    print("=" * 60)
    print("测试 DeepGlobe 数据集加载")
    print("=" * 60)

    data_root = "D:/project/pythonProject/Road_Identification/SAM2-UNet/deepglobe"

    # 测试训练集
    print("\n1. 测试训练集...")
    try:
        train_dataset = DeepGlobeDataset(
            datasets_root=data_root, split="train", data_aug_prob=0.5
        )
        print(f"   训练集样本数: {len(train_dataset)}")

        if len(train_dataset) > 0:
            # 测试读取一个样本
            sample = train_dataset[0]
            print(f"   样本加载成功!")
            print(f"   图像形状: {sample['image'].shape}")
            print(f"   标签形状: {sample['mask'].shape}")
            print(f"   图像名称: {sample['image_name']}")
            print(f"   图像路径: {train_dataset.image_paths[0]}")
            print(f"   标签路径: {train_dataset.mask_paths[0]}")
        print("   训练集测试: 通过 ✓")
    except Exception as e:
        print(f"   训练集测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("测试通过! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_dataset()
