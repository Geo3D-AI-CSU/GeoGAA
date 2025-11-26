"""
地层约束模块
利用地层柱信息为岩性分类添加物理约束
"""
import torch
import torch.nn as nn
import numpy as np

# 地层柱信息
STRATIGRAPHIC_COLUMN = {
    "units": [
        {"name": "T2B2", "rock_unit": 1, "top_interface": 13, "bottom_interface": 12},
        {"name": "T2B1", "rock_unit": 2, "top_interface": 12, "bottom_interface": 11},
        {"name": "T1b", "rock_unit": 3, "top_interface": 11, "bottom_interface": 10},
        {"name": "T1m", "rock_unit": 4, "top_interface": 10, "bottom_interface": 9},
        {"name": "P1m", "rock_unit": 5, "top_interface": 9, "bottom_interface": 8},
        {"name": "P1q", "rock_unit": 6, "top_interface": 8, "bottom_interface": 7},
        {"name": "C3", "rock_unit": 7, "top_interface": 7, "bottom_interface": 6},
        {"name": "C2", "rock_unit": 8, "top_interface": 6, "bottom_interface": 5},
        {"name": "C1", "rock_unit": 9, "top_interface": 5, "bottom_interface": 4},
        {"name": "D3", "rock_unit": 10, "top_interface": 4, "bottom_interface": 3},
        {"name": "D2d", "rock_unit": 11, "top_interface": 3, "bottom_interface": 2},
        {"name": "D1y", "rock_unit": 12, "top_interface": 2, "bottom_interface": 1},
        {"name": "D1n", "rock_unit": 13, "top_interface": 1, "bottom_interface": 0}
    ],
    "interfaces": [
        {"name": "Q-T2B2", "type": "conformable", "age": 13, "level": -1898},
        {"name": "T2B2-T2B1", "type": "conformable", "age": 12, "level": -3734},
        {"name": "T2B1-T1b", "type": "conformable", "age": 11, "level": -4872},
        {"name": "T1b-T1m", "type": "conformable", "age": 10, "level": -6772},
        {"name": "T1m-P1m", "type": "unconformity", "age": 9, "level": -7233},
        {"name": "P1m-P1q", "type": "conformable", "age": 8, "level": -8214},
        {"name": "P1q-C3", "type": "unconformity", "age": 7, "level": -8429},
        {"name": "C3-C2", "type": "conformable", "age": 6, "level": -9145},
        {"name": "C2-C1", "type": "conformable", "age": 5, "level": -9674},
        {"name": "C1-D3", "type": "unconformity", "age": 4, "level": -10602},
        {"name": "D3-D2d", "type": "conformable", "age": 3, "level": -11142},
        {"name": "D2d-D1y", "type": "conformable", "age": 2, "level": -11717},
        {"name": "D1y-D1n", "type": "conformable", "age": 1, "level": -12020}
    ]
}


class StratigraphicConstraint:
    """地层约束类"""

    def __init__(self, stratigraphic_column=STRATIGRAPHIC_COLUMN, min_level_value=-15000):
        self.strat_column = stratigraphic_column
        self.num_units = len(stratigraphic_column["units"])

        # 构建level范围查找表 (rock_unit -> (top_level, bottom_level))
        self.level_ranges = {}
        interface_dict = {ifc["age"]: ifc["level"] for ifc in stratigraphic_column["interfaces"]}

        for unit in stratigraphic_column["units"]:
            rock_unit = unit["rock_unit"]
            top_level = interface_dict[unit["top_interface"]]

            # 🔧 处理最底层地层：bottom_interface=0表示没有底界面
            if unit["bottom_interface"] == 0:
                # 使用一个足够深的虚拟下界
                bottom_level = min_level_value
                print(f"⚠️  注意: 岩性单元 {rock_unit} ({unit['name']}) 是最底层，"
                      f"使用虚拟下界 {min_level_value}")
            else:
                bottom_level = interface_dict[unit["bottom_interface"]]

            self.level_ranges[rock_unit] = (top_level, bottom_level)

        print("\n" + "=" * 80)
        print("📚 地层约束信息初始化")
        print("=" * 80)
        for rock_unit, (top, bottom) in sorted(self.level_ranges.items()):
            unit_name = next(u["name"] for u in stratigraphic_column["units"] if u["rock_unit"] == rock_unit)
            print(f"  岩性 {rock_unit:2d} ({unit_name:6s}): Level范围 [{bottom:7.0f}, {top:7.0f}]")
        print("=" * 80 + "\n")

    def get_level_ranges_tensor(self, device='cuda'):
        """
        返回level范围张量
        返回: [num_units, 2] 张量，第一列是bottom，第二列是top
        """
        ranges = torch.zeros(self.num_units, 2, device=device)
        for rock_unit, (top, bottom) in self.level_ranges.items():
            ranges[rock_unit - 1, 0] = bottom  # 下界
            ranges[rock_unit - 1, 1] = top  # 上界
        return ranges

    def compute_level_compatibility(self, predicted_levels, rock_unit_labels, device='cuda'):
        """
        计算预测level与岩性标签的兼容性

        参数:
        - predicted_levels: [N] 或 [N, 1] 预测的level值
        - rock_unit_labels: [N] 岩性标签 (1-based)

        返回:
        - compatibility: [N] 兼容性分数，1表示完全兼容，0表示不兼容
        """
        # 🔧 修复：确保 predicted_levels 是 1D 张量
        if predicted_levels.dim() > 1:
            predicted_levels = predicted_levels.squeeze()

        N = predicted_levels.shape[0]
        compatibility = torch.ones(N, device=device)

        for rock_unit in range(1, self.num_units + 1):
            mask = (rock_unit_labels == rock_unit)
            if mask.any():
                top_level, bottom_level = self.level_ranges[rock_unit]

                # 计算偏离度
                above_top = torch.clamp(predicted_levels[mask] - top_level, min=0)
                below_bottom = torch.clamp(bottom_level - predicted_levels[mask], min=0)
                deviation = above_top + below_bottom

                # 兼容性: 偏离越大，兼容性越低
                compatibility[mask] = torch.exp(-deviation / 1000.0)  # 1000是尺度参数

        return compatibility

    def get_level_based_prior(self, predicted_levels, device='cuda', temperature=1000.0):
        """
        基于level值计算岩性先验概率

        参数:
        - predicted_levels: [N] 或 [N, 1] 预测的level值
        - temperature: 温度参数，控制prior的锐度

        返回:
        - prior: [N, num_units] 先验概率分布
        """
        # 🔧 修复：确保 predicted_levels 是 1D 张量
        if predicted_levels.dim() > 1:
            predicted_levels = predicted_levels.squeeze()

        N = predicted_levels.shape[0]
        prior = torch.zeros(N, self.num_units, device=device)

        for rock_unit in range(1, self.num_units + 1):
            top_level, bottom_level = self.level_ranges[rock_unit]

            # 计算level在该地层单元范围内的程度
            # 使用高斯核函数
            center = (top_level + bottom_level) / 2
            width = (top_level - bottom_level) / 2

            distance = torch.abs(predicted_levels - center)
            # 🔧 修复：确保赋值的张量是 1D 的
            prior[:, rock_unit - 1] = torch.exp(-distance ** 2 / (2 * (width / 2) ** 2))

        # 归一化
        prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-8)

        return prior


class StratigraphicConstraintLoss(nn.Module):
    """地层约束损失函数"""

    def __init__(self, constraint_module, weight=1.0, temperature=1000.0):
        super().__init__()
        self.constraint = constraint_module
        self.weight = weight
        self.temperature = temperature

    def forward(self, predicted_levels, rock_unit_logits, rock_unit_labels, mask, device='cuda'):
        """
        计算地层约束损失

        参数:
        - predicted_levels: [N] 或 [N, 1] 预测的level值
        - rock_unit_logits: [N, num_classes] 岩性分类logits
        - rock_unit_labels: [N] 真实岩性标签 (1-based)
        - mask: [N] bool掩码
        """
        if not mask.any():
            return torch.tensor(0.0, device=device)

        # 只对有标签的节点计算
        pred_levels_masked = predicted_levels[mask]
        logits_masked = rock_unit_logits[mask]
        labels_masked = rock_unit_labels[mask]

        # 计算基于level的先验分布
        level_prior = self.constraint.get_level_based_prior(
            pred_levels_masked, device=device, temperature=self.temperature
        )

        # 计算模型预测的概率分布
        pred_probs = torch.softmax(logits_masked, dim=1)

        # KL散度损失：鼓励预测分布接近地层先验
        kl_loss = torch.nn.functional.kl_div(
            pred_probs.log(),
            level_prior,
            reduction='batchmean'
        )

        # 添加硬约束：如果预测的岩性与level完全不兼容，增加惩罚
        compatibility = self.constraint.compute_level_compatibility(
            pred_levels_masked, labels_masked, device=device
        )
        compatibility_loss = (1 - compatibility).mean()

        total_loss = kl_loss + compatibility_loss

        return self.weight * total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # alpha可以是标量或类别权重向量
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha] * 13)
            else:
                # 🔧 修复：使用 clone().detach() 而不是 torch.tensor()
                if isinstance(alpha, torch.Tensor):
                    self.alpha = alpha.clone().detach()
                else:
                    self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        参数:
        - inputs: [N, num_classes] logits
        - targets: [N] 类别标签 (0-based)
        """
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(rock_unit_labels, mask, num_classes=13):
    """
    计算类别权重（用于处理不平衡）

    参数:
    - rock_unit_labels: [N] 岩性标签 (1-based)
    - mask: [N] bool掩码
    - num_classes: 类别数

    返回:
    - weights: [num_classes] 类别权重
    """
    labels_masked = rock_unit_labels[mask].cpu().numpy()

    # 统计每个类别的样本数（转换为0-based）
    class_counts = np.bincount(labels_masked - 1, minlength=num_classes)

    # 避免除零
    class_counts = np.maximum(class_counts, 1)

    # 计算权重：样本少的类别权重大
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)

    # 归一化
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def post_process_with_level_constraint(predicted_levels, rock_unit_logits,
                                       constraint_module, device='cuda'):
    """
    后处理：根据level约束修正岩性预测

    参数:
    - predicted_levels: [N] 或 [N, 1] 预测的level值
    - rock_unit_logits: [N, num_classes] 岩性分类logits
    - constraint_module: StratigraphicConstraint实例

    返回:
    - corrected_predictions: [N] 修正后的岩性预测 (0-based)
    """
    # 🔧 修复：确保 predicted_levels 是 1D 张量
    if predicted_levels.dim() > 1:
        predicted_levels = predicted_levels.squeeze()

    N = predicted_levels.shape[0]

    # 原始预测
    original_preds = torch.argmax(rock_unit_logits, dim=1)

    # 计算每个节点对所有岩性单元的兼容性
    level_prior = constraint_module.get_level_based_prior(
        predicted_levels, device=device, temperature=500.0
    )

    # 模型预测概率
    pred_probs = torch.softmax(rock_unit_logits, dim=1)

    # 融合先验和模型预测（加权平均）
    alpha = 0.3  # 先验权重
    combined_probs = alpha * level_prior + (1 - alpha) * pred_probs

    # 基于融合概率做最终预测
    corrected_preds = torch.argmax(combined_probs, dim=1)

    return corrected_preds


if __name__ == "__main__":
    # 测试代码
    print("测试地层约束模块")

    constraint = StratigraphicConstraint()

    # 模拟数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 1000
    predicted_levels = torch.randn(N, device=device) * 3000 - 7000  # 模拟level值
    rock_unit_labels = torch.randint(1, 14, (N,), device=device)  # 1-13

    # 测试兼容性计算
    compatibility = constraint.compute_level_compatibility(
        predicted_levels, rock_unit_labels, device=device
    )
    print(f"\n平均兼容性: {compatibility.mean():.4f}")

    # 测试先验计算
    level_prior = constraint.get_level_based_prior(predicted_levels, device=device)
    print(f"先验分布形状: {level_prior.shape}")
    print(f"先验分布和: {level_prior[0].sum():.4f}")

    # 测试损失函数
    rock_unit_logits = torch.randn(N, 13, device=device)
    mask = torch.ones(N, dtype=torch.bool, device=device)

    loss_fn = StratigraphicConstraintLoss(constraint, weight=1.0)
    loss = loss_fn(predicted_levels, rock_unit_logits, rock_unit_labels, mask, device)
    print(f"\n地层约束损失: {loss.item():.4f}")

    # 测试Focal Loss
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    focal_loss = focal_loss_fn(rock_unit_logits, rock_unit_labels - 1)
    print(f"Focal Loss: {focal_loss.item():.4f}")

    print("\n✅ 测试完成")