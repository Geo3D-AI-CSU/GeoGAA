import torch
import random
import numpy as np

def select_device(desired_gpu=2):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"可用的 GPU 数量: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        if desired_gpu < num_gpus:
            # 使用 torch.cuda.set_device 来设置默认 GPU
            torch.cuda.set_device(desired_gpu)
            # 返回设备对象
            device = torch.device(f'cuda:{desired_gpu}')
            print(f"使用设备: {device}")
        else:
            print(f"GPU {desired_gpu} 不可用。使用 CPU。")
            device = torch.device('cpu')
    else:
        print("CUDA 不可用。使用 CPU。")
        device = torch.device('cpu')

    return device

def set_random_seed(seed=42):
    """
    设置随机种子以确保结果的可重复性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False