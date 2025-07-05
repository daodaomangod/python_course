import torch

# 判断是否可以用GPU

cuda_available = torch.cuda.is_available()
map_available = torch.backends.mps.is_available()  # 检查MPS的可用性
if cuda_available:
    device = 'cuda'
elif map_available:
    device = 'mps'  # mps是苹果m系列芯片的GPU
else:
    device = 'cpu'
print(f'Using device is {device}')
