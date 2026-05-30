import sys
import subprocess

# 检查 Python 版本
print("=" * 50)
print("1. Python 版本检查")
print("=" * 50)
python_version = sys.version_info
print(f"当前 Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version >= (3, 8):
    print("✅ Python 版本满足要求 (>=3.8)")
else:
    print("❌ Python 版本不满足要求，需要 >=3.8")

# 检查 PyTorch
print("\n" + "=" * 50)
print("2. PyTorch 版本检查")
print("=" * 50)
try:
    import torch

    print(f"PyTorch 版本: {torch.__version__}")

    # 解析版本号
    torch_version = torch.__version__.split('+')[0]  # 去掉 CUDA 版本后缀
    version_parts = torch_version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])

    if major > 1 or (major == 1 and minor >= 7):
        print("✅ PyTorch 版本满足要求 (>=1.7)")
    else:
        print("❌ PyTorch 版本不满足要求，需要 >=1.7")

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ CUDA 不可用，PyTorch 正在使用 CPU 模式")
        print("   建议安装带 CUDA 支持的 PyTorch 以获得更好性能")

except ImportError:
    print("❌ PyTorch 未安装")

# 检查 TorchVision
print("\n" + "=" * 50)
print("3. TorchVision 版本检查")
print("=" * 50)
try:
    import torchvision

    print(f"TorchVision 版本: {torchvision.__version__}")

    # 解析版本号
    tv_version = torchvision.__version__.split('+')[0]
    version_parts = tv_version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])

    if major > 0 or (major == 0 and minor >= 8):
        print("✅ TorchVision 版本满足要求 (>=0.8)")
    else:
        print("❌ TorchVision 版本不满足要求，需要 >=0.8")

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("✅ TorchVision 可以使用 CUDA 加速")
    else:
        print("⚠️ TorchVision 正在使用 CPU 模式")

except ImportError:
    print("❌ TorchVision 未安装")

# 总结
print("\n" + "=" * 50)
print("4. 环境检查总结")
print("=" * 50)
