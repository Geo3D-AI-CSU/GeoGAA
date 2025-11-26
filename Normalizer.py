import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Normalizer:
    def __init__(self):
        self.level_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.coord_scaler = MinMaxScaler(feature_range=(-1, 1))

    def fit_transform_level_masked(self, level_masked):
        """
        仅对掩码节点的 level 进行归一化，并拟合 scaler。
        """
        level_np = level_masked.cpu().numpy().reshape(-1, 1)
        self.level_scaler.fit(level_np)
        level_norm = self.level_scaler.transform(level_np)
        return torch.tensor(level_norm.squeeze(), dtype=torch.float32).to(
            level_masked.device)  # 修改: 使用 squeeze() 使其成为一维张量

    def inverse_transform_level(self, level_norm):
        """
        🔧 修复：
        1. 添加 .detach() 以避免梯度错误
        2. 使用 .squeeze() 确保返回1D张量
        """
        level_np = level_norm.detach().cpu().numpy().reshape(-1, 1)
        level_original = self.level_scaler.inverse_transform(level_np)
        # 返回1D张量，避免维度不匹配
        return torch.tensor(level_original.squeeze(), dtype=torch.float32).to(level_norm.device)

    def fit_transform_coords(self, coords):
        coords_np = coords.cpu().numpy()
        coords_norm = self.coord_scaler.fit_transform(coords_np)
        return torch.tensor(coords_norm, dtype=torch.float32).to(coords.device)

    def inverse_transform_coords(self, coords_norm):
        coords_np = coords_norm.cpu().numpy()
        coords_original = self.coord_scaler.inverse_transform(coords_np)
        return torch.tensor(coords_original, dtype=torch.float32).to(coords_norm.device)

    def fit_transform_values(self, min_values, max_values):
        """
        对 min_values 和 max_values 进行归一化到 [0, 1] 区间。
        忽略 -9999 的值，不对其进行归一化。
        """
        # 转换为numpy数组进行处理
        min_values = np.array(min_values)
        max_values = np.array(max_values)

        # 归一化处理，使用有效数据（忽略 -9999 的值）
        valid_mask_min = min_values != -9999  # 标记有效的 min_values
        valid_mask_max = max_values != -9999  # 标记有效的 max_values

        # 归一化 min_values 和 max_values，仅对有效值进行归一化
        min_values_norm = np.copy(min_values)
        max_values_norm = np.copy(max_values)
        max_values_norm = max_values_norm.astype(np.float64)
        min_values_norm = min_values_norm.astype(np.float64)
        # 仅对有效值部分进行归一化
        if np.any(valid_mask_min):  # 确保有效的min值部分存在
            valid_min_values = min_values[valid_mask_min]  # 获取有效部分
            min_norm = (valid_min_values - valid_min_values.min()) / (
                        valid_min_values.max() - valid_min_values.min())  # 归一化有效部分
            min_values_norm[valid_mask_min] = min_norm  # 将归一化结果赋值回

        if np.any(valid_mask_max):  # 确保有效的max值部分存在
            valid_max_values = max_values[valid_mask_max]  # 获取有效部分
            max_norm = (valid_max_values - valid_max_values.min()) / (
                        valid_max_values.max() - valid_max_values.min())  # 归一化有效部分
            max_values_norm[valid_mask_max] = max_norm  # 将归一化结果赋值回

        return min_values_norm, max_values_norm

    def inverse_transform_values(self, min_values_norm, max_values_norm, min_values, max_values):
        """
        对 min_values 和 max_values 进行反归一化，恢复到原始范围。
        忽略 -9999 的值，不对其进行反归一化。
        """
        # 转换为numpy数组进行处理
        min_values_norm = np.array(min_values_norm)
        max_values_norm = np.array(max_values_norm)
        min_values = np.array(min_values)
        max_values = np.array(max_values)

        # 过滤掉 -9999 的值
        valid_mask = (min_values_norm != -9999) & (max_values_norm != -9999)

        # 反归一化
        min_values_original = np.copy(min_values_norm)
        max_values_original = np.copy(max_values_norm)

        min_values_original[valid_mask] = min_values_norm[valid_mask] * (max_values[valid_mask] - min_values[valid_mask]) + min_values[valid_mask]
        max_values_original[valid_mask] = max_values_norm[valid_mask] * (max_values[valid_mask] - min_values[valid_mask]) + min_values[valid_mask]

        return min_values_original, max_values_original