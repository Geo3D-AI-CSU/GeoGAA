import torch
import torch.nn.functional as F


def scalar_loss_slow(predicted_scalar, rock_unit_labels, min_values, max_values):
    """
    计算岩性标量场的损失函数，确保预测的标量值在指定的范围内，处理缺失的最小值和最大值。

    参数：
    - predicted_scalar (Tensor): 预测的标量场值，形状为 (num_nodes, )。
    - rock_unit_labels (Tensor): 岩性标签，形状为 (num_nodes, )，用于查找对应的范围。
    - min_values (Tensor or None): 每个岩性标签对应的最小标量场值，形状为 (num_labels, )，可以为 None。
    - max_values (Tensor or None): 每个岩性标签对应的最大标量场值，形状为 (num_labels, )，可以为 None。

    返回：
    - loss (Tensor): 计算得到的损失值。
    """

    # 获取岩性标签对应的最小值和最大值
    min_values_for_labels = min_values[rock_unit_labels] if min_values != -9999 else -9999
    max_values_for_labels = max_values[rock_unit_labels] if max_values != -9999 else -9999

    # 将min_values和max_values在设备上与predicted_scalar保持一致
    if min_values_for_labels is not -9999:
        min_values_for_labels = min_values_for_labels.to(predicted_scalar.device)
    if max_values_for_labels is not -9999:
        max_values_for_labels = max_values_for_labels.to(predicted_scalar.device)

    # 初始化损失值
    loss = torch.zeros_like(predicted_scalar,requires_grad=True)

    # 遍历每个节点并计算损失
    for i in range(len(predicted_scalar)):
        min_value = min_values_for_labels[i]
        max_value = max_values_for_labels[i]
        scalar_value = predicted_scalar[i]

        # 判断 min_value 和 max_value 的值并计算损失
        if min_value == -9999 and max_value == -9999:
            continue  # 如果两个都为 -9999，跳过该节点的损失计算

        if min_value == -9999:
            # 如果 min_value 是 -9999，损失为 |predicted_scalar - max_value|，当 predicted_scalar > max_value 时
            loss = torch.where(scalar_value > max_value, torch.abs(scalar_value - max_value), loss)
        elif max_value == -9999:
            # 如果 max_value 是 -9999，损失为 |predicted_scalar - min_value|，当 predicted_scalar < min_value 时
            loss = torch.where(scalar_value < min_value, torch.abs(scalar_value - min_value), loss)
        else:
            # 如果 min_value 和 max_value 都不为 -9999，分别计算两种情况的损失
            loss = torch.where(scalar_value > max_value, torch.abs(scalar_value - max_value), loss)
            loss = torch.where(scalar_value < min_value, torch.abs(scalar_value - min_value), loss)

    return loss.sum()  # 返回损失值


def scalar_loss(predicted_scalar, rock_unit_labels, min_values, max_values):
    """
    计算岩性标量场的损失函数，确保预测的标量值在指定的范围内，处理缺失的最小值和最大值。

    参数：
    - predicted_scalar (Tensor): 预测的标量场值，形状为 (num_nodes, )。
    - rock_unit_labels (Tensor): 岩性标签，形状为 (num_nodes, )，用于查找对应的范围。
    - min_values (Tensor or None): 每个岩性标签对应的最小标量场值，形状为 (num_labels, )，可以为 None。
    - max_values (Tensor or None): 每个岩性标签对应的最大标量场值，形状为 (num_labels, )，可以为 None。

    返回：
    - loss (Tensor): 计算得到的损失值。
    """

    # 获取岩性标签对应的最小值和最大值
    min_values_for_labels = min_values[rock_unit_labels-1]  # 获取每个岩性标签对应的最小值
    max_values_for_labels = max_values[rock_unit_labels-1]  # 获取每个岩性标签对应的最大值

    # 将min_values和max_values在设备上与predicted_scalar保持一致
    min_values_for_labels = min_values_for_labels.to(predicted_scalar.device)
    max_values_for_labels = max_values_for_labels.to(predicted_scalar.device)

    # 初始化损失值
    loss = torch.zeros_like(predicted_scalar, requires_grad=True)

    # 计算标量值的损失
    min_mask = min_values_for_labels != -9999
    max_mask = max_values_for_labels != -9999

    # 处理 min_value 为 -9999 的情况
    loss = torch.where(~min_mask & max_mask & (predicted_scalar > max_values_for_labels),
                        torch.abs(predicted_scalar - max_values_for_labels),
                        loss)

    # 处理 max_value 为 -9999 的情况
    loss = torch.where(~max_mask & min_mask & (predicted_scalar < min_values_for_labels),
                        torch.abs(predicted_scalar - min_values_for_labels),
                        loss)

    # 处理 min_value 和 max_value 都不为 -9999 的情况
    loss = torch.where(min_mask & max_mask & (predicted_scalar > max_values_for_labels),
                        torch.abs(predicted_scalar - max_values_for_labels),
                        loss)
    loss = torch.where(min_mask & max_mask & (predicted_scalar < min_values_for_labels),
                        torch.abs(predicted_scalar - min_values_for_labels),
                        loss)
    # 返回所有节点的损失值之和
    return loss.sum()


# 定义损失函数
def level_loss(predicted, target):
    return  F.mse_loss(predicted, target)

def rock_unit_loss(predicted_rock_units, rock_unit_labels):
    rock_unit_labels = rock_unit_labels - 1
    return F.cross_entropy(predicted_rock_units, rock_unit_labels)

def gradient_loss(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    计算梯度损失。
    :param predicted_levels: 预测的level值
    :param coords: 节点坐标
    :param dx, dy, dz: 真实梯度（已知）
    :param edge_index: 图的边索引
    :param mask_gradient: 掩码，指定哪些节点需要计算梯度
    :return: 梯度损失
    mask_sample: 用于指示哪些节点需要计算梯度,有邻居节点的节点才计算梯度
    """
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    mask_sample = torch.zeros(mask_indices .size(0), dtype=torch.bool)  # 默认全为 False，表示没有邻居
    row, col = edge_index
    gradients = []
    # 确保 dx, dy, dz 有梯度支持
    # dx.requires_grad_(True)
    # dy.requires_grad_(True)
    # dz.requires_grad_(True)

    # 遍历每个节点，检查是否有邻居，有的话给它设置成True
    # 提前计算 row == node 和 col == node 的布尔数组，减少重复计算
    for i, node in enumerate(mask_indices):
        row_mask = (row == node)  # 记录 row 中是否等于 node
        col_mask = (col == node)  # 记录 col 中是否等于 node

        # 查找 row 和 col 中与当前 node 相关的邻居
        col_neighbors = col[row_mask]  # row 中的邻居
        row_neighbors = row[col_mask]  # col 中的邻居

        # 合并 row 和 col 中的邻居
        neighbors = torch.cat((row_neighbors, col_neighbors))

        # 更新掩码，检查是否有邻居
        if neighbors.numel() > 0:
            mask_sample[i] = True  # 有邻居，掩码为 True
        else:
            mask_sample[i] = False  # 没有邻居，掩码为 False

        # 跳过没有邻居的节点
        if not mask_sample[i]:
            gradients.append(torch.zeros(3, device=predicted_levels.device))  # 无邻居节点梯度为 [0, 0, 0]
            continue  # 跳过没有邻居的节点

        neighbors_v = neighbors
        # 计算delta_coords和delta_levels
        delta_coords = coords[neighbors_v] - coords[node].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[node].unsqueeze(0)

        # 计算梯度（使用自动微分）
        delta_coords.requires_grad_(True)
        delta_levels.requires_grad_(True)

        # 计算雅可比矩阵的转置与delta_coords相乘
        AtA = torch.matmul(delta_coords.T, delta_coords)
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))

        # 使用torch.linalg.pinv来稳定求解
        try:
            gradient = torch.linalg.pinv(AtA) @ Atb
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            print(f"Solving fail, setting gradient to [0, 0, 0].")

    # 计算完所有节点的梯度后，将它们堆叠成一个张量
    gradient_estimates = torch.stack(gradients) if gradients else torch.zeros_like(dx)  # 如果没有有效的梯度，返回零张量

    # 真实梯度
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # 规范化梯度
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients
    mask_sample = mask_sample.to(normalized_true_gradients.device)
    # 计算掩膜下的梯度损失
    masked_predicted_gradients = normalized_predicted_gradients * mask_sample.unsqueeze(-1)  # 应用掩膜
    masked_true_gradients = normalized_true_gradients * mask_sample.unsqueeze(-1)  # 应用掩膜

    # 计算预测梯度与真实梯度的余弦相似度
    cos_theta = (masked_predicted_gradients * masked_true_gradients).sum(dim=-1)
    mask = (cos_theta != 0)
    # 计算角度损失
    angle_loss = torch.mean(1 - cos_theta[mask])

    return angle_loss


def gradient_loss_slow(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    计算梯度损失。
    :param predicted_levels: 预测的level值
    :param coords: 节点坐标
    :param dx, dy, dz: 真实梯度（已知）
    :param edge_index: 图的边索引
    :param mask_gradient: 掩码，指定哪些节点需要计算梯度
    :return: 梯度损失
    mask_sample: 用于指示哪些节点需要计算梯度,有邻居节点的节点才计算梯度
    """
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    mask_sample = torch.zeros(mask_indices .size(0), dtype=torch.bool)  # 默认全为 False，表示没有邻居
    row, col = edge_index
    gradients = []
    # 确保 dx, dy, dz 有梯度支持
    # dx.requires_grad_(True)
    # dy.requires_grad_(True)
    # dz.requires_grad_(True)

    # 遍历每个节点，检查是否有邻居，有的话给它设置成True
    for i, node in enumerate(mask_indices):
        # 检查 node 是否在 row 或 col 中
        if (row == node).any() and (col == node).any():
            neighbors = torch.cat((col[row == node], row[col == node]))  # 合并在 row 和 col 中的邻居
        elif (row == node).any():
            neighbors = col[row == node]
        elif (col == node).any():
            neighbors = row[col == node]
        if neighbors.numel() > 0:
            mask_sample[i] = True  # 有邻居，掩码为True
        else:
            mask_sample[i] = False  # 没有邻居，掩码为False
        if not mask_sample[i]:  # 如果该节点没有邻居（掩码为 False），跳过梯度计算
            # print(f"Node {v} has no neighbors, setting gradient to [0, 0, 0].")
            gradients.append(torch.zeros(3, device=predicted_levels.device))  # 没有邻居的节点梯度为 [0, 0, 0]
            continue  # 跳过没有邻居的节点
        neighbors_v = neighbors
        # 计算delta_coords和delta_levels
        delta_coords = coords[neighbors_v] - coords[node].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[node].unsqueeze(0)

        # 计算梯度（使用自动微分）
        delta_coords.requires_grad_(True)
        delta_levels.requires_grad_(True)

        # 计算雅可比矩阵的转置与delta_coords相乘
        AtA = torch.matmul(delta_coords.T, delta_coords)
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))

        # 使用torch.linalg.pinv来稳定求解
        try:
            gradient = torch.linalg.pinv(AtA) @ Atb
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            print(f"Solving fail, setting gradient to [0, 0, 0].")

    # 计算完所有节点的梯度后，将它们堆叠成一个张量
    gradient_estimates = torch.stack(gradients) if gradients else torch.zeros_like(dx)  # 如果没有有效的梯度，返回零张量

    # 真实梯度
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # 规范化梯度
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients
    mask_sample = mask_sample.to(normalized_true_gradients.device)
    # 计算掩膜下的梯度损失
    masked_predicted_gradients = normalized_predicted_gradients * mask_sample.unsqueeze(-1)  # 应用掩膜
    masked_true_gradients = normalized_true_gradients * mask_sample.unsqueeze(-1)  # 应用掩膜

    # 计算预测梯度与真实梯度的余弦相似度
    cos_theta = (masked_predicted_gradients * masked_true_gradients).sum(dim=-1)
    mask = (cos_theta != 0)
    # 计算角度损失
    angle_loss = torch.sum(1 - cos_theta[mask])

    return angle_loss


def gradient_loss_old(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    # 获取需要计算梯度的节点索引
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    num_masked = mask_indices.size(0)

    # 分离边的来源和目标
    row, col = edge_index

    # 准备一个列表来存储每个节点的梯度
    gradients = []

    for v in mask_indices:
        # 获取节点v的邻居节点
        neighbors_v = col[row == v]

        if neighbors_v.numel() == 0:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            continue

        # 计算坐标和level的差异
        delta_coords = coords[neighbors_v] - coords[v].unsqueeze(0)  # [num_neighbors, 3]
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[v].unsqueeze(0)  # [num_neighbors]

        # 最小二乘法求解梯度
        AtA = torch.matmul(delta_coords.T, delta_coords)  # [3, 3]
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))  # [3, 1]

        # 计算梯度
        try:
            # 使用伪逆避免非正定矩阵问题
            gradient = torch.linalg.pinv(AtA) @ Atb  # [3, 1]
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))

    # 合并所有梯度
    gradient_estimates = torch.stack(gradients)  # [num_masked, 3]

    # 计算真实梯度
    true_gradients = torch.stack((dx, dy, dz), dim=-1)  # [num_masked, 3]

    # 规范化梯度
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients

    # 计算预测梯度与真实梯度的余弦相似度
    cos_theta = (normalized_predicted_gradients * normalized_true_gradients).sum(dim=-1)
    angle_loss = torch.sum(1 - cos_theta)

    return angle_loss



def gradient_loss_autogradient(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    计算预测的 `level` 对应坐标的梯度，并与目标梯度 (dx, dy, dz) 进行对比，计算角度损失。
    """
    # mask_gradient中为True的位置需要计算梯度
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    row, col = edge_index  # 获取边的连接信息

    gradients = []

    # 遍历所有需要计算梯度的节点
    for v in mask_indices:
        neighbors_v = col[row == v]
        if neighbors_v.numel() == 0:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            continue

        # 计算delta_coords和delta_levels
        delta_coords = coords[neighbors_v] - coords[v].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[v].unsqueeze(0)

        # 使用自动微分计算预测的梯度
        coords_v = coords[v].unsqueeze(0).requires_grad_(True)  # 设置为可求导
        predicted_levels_v = predicted_levels[v].unsqueeze(0).requires_grad_(True)  # 设置为可求导

        # 计算损失（例如，预测值与目标值之间的均方误差）
        loss = torch.abs(predicted_levels_v - predicted_levels[neighbors_v])

        # 清空梯度，以避免在计算其他梯度时受到影响
        coords_v.grad = None  # 确保清空之前的梯度

        # 反向传播，计算梯度
        loss.backward()

        # 获取节点v的梯度
        gradient = coords_v.grad
        gradients.append(gradient.squeeze())

    # 计算完所有节点的梯度后，将它们堆叠成一个张量
    gradient_estimates = torch.stack(gradients)

    # 真实梯度
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # 规范化梯度
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients

    # 计算预测梯度与真实梯度的余弦相似度
    cos_theta = (normalized_predicted_gradients * normalized_true_gradients).sum(dim=-1)

    # 计算角度损失
    angle_loss = torch.sum(1 - cos_theta)

    return angle_loss
