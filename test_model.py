import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from model.unet import UNet


def test_model():
    print("=" * 60)
    print("测试 UNet 模型")
    print("=" * 60)

    # 创建模型
    print("\n创建 UNet 模型...")
    model = UNet(block="BasicBlock")
    print("模型创建成功!")

    # 打印模型结构
    print(f"\n模型结构:")
    print(model)

    # 测试前向传播
    print("\n测试前向传播...")
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 检查输出通道
    if output.shape[1] == 2:
        print("输出 2 通道 - 正常 (需要取第 0 通道用于道路分割)")
    else:
        print(f"警告: 输出通道数不是 2，而是 {output.shape[1]}")

    print("\n" + "=" * 60)
    print("模型测试完成! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
