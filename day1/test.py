import torch

print("PyTorch 版本:", torch.__version__)

print("CUDA 是否可用:", torch.cuda.is_available())
print("可用 GPU 数量:", torch.cuda.device_count())
print("当前 GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")

# 创建 CPU 张量
x = torch.tensor([1.0, 2.0, 3.0])
print("CPU 张量:", x)

# 如果 CUDA 可用，将张量移至 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = x.to(device)
    print("GPU 张量:", y)

    # 在 GPU 上执行计算
    z = y + 2
    print("计算结果:", z)

    # 将结果转回 CPU
    print("转回 CPU:", z.to("cpu"))

    print("PyTorch 编译 CUDA 版本:", torch.version.cuda)
    print("系统 CUDA 版本:", torch.version.cuda if torch.cuda.is_available() else "仅 CPU")