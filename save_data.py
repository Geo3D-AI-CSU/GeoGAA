import pandas as pd
import torch
import numpy as np
import os

def save_fault_result_to_csv(graph_data, predicted_level, nodes,suffix=''):
    """
    保存所有节点的 X, Y, Z 坐标和预测的 Level 值到 CSV 文件中。

    参数：
    - graph_data: 包含图数据的对象，必须包含 original_coords 属性。
    - predicted_level: 模型预测的 Level 值，应该是 torch.Tensor 或 numpy.ndarray。
    - nodes: 要保存的节点索引，应该是 torch.Tensor, numpy.ndarray 或 list。
    - suffix: 文件名后缀，用于区分不同的 Level 组。
    """

    # 转换 predicted_level 为 numpy 数组
    if isinstance(predicted_level, torch.Tensor):
        predicted_level_np = predicted_level.detach().cpu().numpy()
    elif isinstance(predicted_level, np.ndarray):
        predicted_level_np = predicted_level
    else:
        raise TypeError("predicted_level 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 original_coords 为 numpy 数组
    if isinstance(graph_data.original_coords, torch.Tensor):
        original_coords_np = graph_data.original_coords.detach().cpu().numpy()
    elif isinstance(graph_data.original_coords, np.ndarray):
        original_coords_np = graph_data.original_coords
    else:
        raise TypeError("graph_data.original_coords 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 nodes 为 numpy 数组
    if isinstance(nodes, torch.Tensor):
        nodes_np = nodes.detach().cpu().numpy()
    elif isinstance(nodes, np.ndarray):
        nodes_np = nodes
    elif isinstance(nodes, list):
        nodes_np = np.array(nodes)
    else:
        nodes_np = np.array(nodes)

    # 打印调试信息
    print(f"predicted_level_np 类型: {type(predicted_level_np)}, shape: {predicted_level_np.shape}")
    print(f"original_coords_np 类型: {type(original_coords_np)}, shape: {original_coords_np.shape}")
    print(f"nodes_np 类型: {type(nodes_np)}, shape: {nodes_np.shape}")

    # 确保 nodes_np 是整数类型，用于索引
    nodes_np = nodes_np.astype(int)

    # 检查 nodes_np 是否超出范围
    if nodes_np.size > 0:
        if nodes_np.max() >= len(predicted_level_np):
            raise IndexError("nodes 中存在超出 predicted_level 范围的索引")
        if nodes_np.max() >= len(original_coords_np):
            raise IndexError("nodes 中存在超出 original_coords 范围的索引")

    # 构建数据字典
    try:
        data = {
            'node': nodes_np,
            'predicted_level': predicted_level_np[nodes_np],
            'original_coords_x': original_coords_np[nodes_np, 0],
            'original_coords_y': original_coords_np[nodes_np, 1],
            'original_coords_z': original_coords_np[nodes_np, 2]
        }
    except Exception as e:
        print("构建数据字典时出错:", e)
        print(f"nodes_np: {nodes_np}")
        print(f"predicted_level_np: {predicted_level_np}")
        print(f"original_coords_np: {original_coords_np}")
        raise e

    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    csv_filename = f'training_results{suffix}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"预测结果已保存到 {csv_filename}")

def save_horizon_result_to_csv(graph_data, predicted_level, fault_features, nodes, suffix=''):
    """
    保存所有节点的 X, Y, Z 坐标、预测的 Level 值和断层特征到 CSV 文件中。

    参数：
    - graph_data: 包含图数据的对象，必须包含 original_coords 属性。
    - predicted_level: 模型预测的 Level 值，应该是 torch.Tensor 或 numpy.ndarray。
    - fault_features: 每个节点的断层特征，应该是 torch.Tensor 或 numpy.ndarray。
    - nodes: 要保存的节点索引，应该是 torch.Tensor, numpy.ndarray 或 list。
    - suffix: 文件名后缀，用于区分不同的 Level 组。
    """

    # 转换 predicted_level 为 numpy 数组
    if isinstance(predicted_level, torch.Tensor):
        predicted_level_np = predicted_level.detach().cpu().numpy()
    elif isinstance(predicted_level, np.ndarray):
        predicted_level_np = predicted_level
    else:
        raise TypeError("predicted_level 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 fault_features 为 numpy 数组
    if isinstance(fault_features, torch.Tensor):
        fault_features_np = fault_features.detach().cpu().numpy()
    elif isinstance(fault_features, np.ndarray):
        fault_features_np = fault_features
    else:
        raise TypeError("fault_features 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 original_coords 为 numpy 数组
    if isinstance(graph_data.original_coords, torch.Tensor):
        original_coords_np = graph_data.original_coords.detach().cpu().numpy()
    elif isinstance(graph_data.original_coords, np.ndarray):
        original_coords_np = graph_data.original_coords
    else:
        raise TypeError("graph_data.original_coords 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 nodes 为 numpy 数组
    if isinstance(nodes, torch.Tensor):
        nodes_np = nodes.detach().cpu().numpy()
    elif isinstance(nodes, np.ndarray):
        nodes_np = nodes
    elif isinstance(nodes, list):
        nodes_np = np.array(nodes)
    else:
        nodes_np = np.array(nodes)

    # 打印调试信息
    print(f"predicted_level_np 类型: {type(predicted_level_np)}, shape: {predicted_level_np.shape}")
    print(f"fault_features_np 类型: {type(fault_features_np)}, shape: {fault_features_np.shape}")
    print(f"original_coords_np 类型: {type(original_coords_np)}, shape: {original_coords_np.shape}")
    print(f"nodes_np 类型: {type(nodes_np)}, shape: {nodes_np.shape}")

    # 确保 nodes_np 是整数类型，用于索引
    nodes_np = nodes_np.astype(int)

    # 检查 nodes_np 是否超出范围
    if nodes_np.size > 0:
        if nodes_np.max() >= len(predicted_level_np):
            raise IndexError("nodes 中存在超出 predicted_level 范围的索引")
        if nodes_np.max() >= len(original_coords_np):
            raise IndexError("nodes 中存在超出 original_coords 范围的索引")
        if fault_features_np.shape[0] != len(original_coords_np):
            raise ValueError("fault_features 的节点数与 original_coords 不匹配")

    # 构建数据字典
    try:
        data = {
            'node': nodes_np,
            'predicted_level': predicted_level_np[nodes_np].squeeze(),
            'original_coords_x': original_coords_np[nodes_np, 0],
            'original_coords_y': original_coords_np[nodes_np, 1],
            'original_coords_z': original_coords_np[nodes_np, 2],
        }

        # 添加断层特征到数据字典
        num_faults = fault_features_np.shape[1]
        for i in range(num_faults):
            data[f'fault_feature_{i}'] = fault_features_np[nodes_np, i]

    except Exception as e:
        print("构建数据字典时出错:", e)
        print(f"nodes_np: {nodes_np}")
        print(f"predicted_level_np: {predicted_level_np}")
        print(f"original_coords_np: {original_coords_np}")
        print(f"fault_features_np: {fault_features_np}")
        raise e

    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    csv_filename = f'horizon_training_results{suffix}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"预测结果已保存到 {csv_filename}")
def save_rock_result_to_csv(graph_data, predicted_level, fault_features, nodes, predicted_rock_units, suffix='',result_dir=None):
    """
    保存所有节点的 X, Y, Z 坐标、预测的 Level 值和断层特征到 CSV 文件中。

    参数：
    - graph_data: 包含图数据的对象，必须包含 original_coords 属性。
    - predicted_level: 模型预测的 Level 值，应该是 torch.Tensor 或 numpy.ndarray。
    - fault_features: 每个节点的断层特征，应该是 torch.Tensor 或 numpy.ndarray。
    - nodes: 要保存的节点索引，应该是 torch.Tensor, numpy.ndarray 或 list。
    - suffix: 文件名后缀，用于区分不同的 Level 组。
    """

    # 转换 predicted_level 为 numpy 数组
    if isinstance(predicted_level, torch.Tensor):
        predicted_level_np = predicted_level.detach().cpu().numpy()
    elif isinstance(predicted_level, np.ndarray):
        predicted_level_np = predicted_level
    else:
        raise TypeError("predicted_level 必须是 torch.Tensor 或 numpy.ndarray 类型")

    if isinstance(predicted_rock_units, torch.Tensor):
        predicted_rock_units_np = predicted_rock_units.detach().cpu().numpy()
    elif isinstance(predicted_rock_units, np.ndarray):
        predicted_rock_units_np = predicted_rock_units
    else:
        raise TypeError("predicted_level 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 fault_features 为 numpy 数组
    if isinstance(fault_features, torch.Tensor):
        fault_features_np = fault_features.detach().cpu().numpy()
    elif isinstance(fault_features, np.ndarray):
        fault_features_np = fault_features
    else:
        raise TypeError("fault_features 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 original_coords 为 numpy 数组
    if isinstance(graph_data.original_coords, torch.Tensor):
        original_coords_np = graph_data.original_coords.detach().cpu().numpy()
    elif isinstance(graph_data.original_coords, np.ndarray):
        original_coords_np = graph_data.original_coords
    else:
        raise TypeError("graph_data.original_coords 必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 转换 nodes 为 numpy 数组
    if isinstance(nodes, torch.Tensor):
        nodes_np = nodes.detach().cpu().numpy()
    elif isinstance(nodes, np.ndarray):
        nodes_np = nodes
    elif isinstance(nodes, list):
        nodes_np = np.array(nodes)
    else:
        nodes_np = np.array(nodes)

    # 打印调试信息
    print(f"predicted_level_np 类型: {type(predicted_level_np)}, shape: {predicted_level_np.shape}")
    print(f"predicted_rock_units_np 类型:{type(predicted_rock_units_np)}, shape: {predicted_rock_units_np.shape}")
    print(f"fault_features_np 类型: {type(fault_features_np)}, shape: {fault_features_np.shape}")
    print(f"original_coords_np 类型: {type(original_coords_np)}, shape: {original_coords_np.shape}")
    print(f"nodes_np 类型: {type(nodes_np)}, shape: {nodes_np.shape}")

    # 确保 nodes_np 是整数类型，用于索引
    nodes_np = nodes_np.astype(int)

    # 检查 nodes_np 是否超出范围
    if nodes_np.size > 0:
        if nodes_np.max() >= len(predicted_level_np):
            raise IndexError("nodes 中存在超出 predicted_level 范围的索引")
        if nodes_np.max() >= len(original_coords_np):
            raise IndexError("nodes 中存在超出 original_coords 范围的索引")
        if fault_features_np.shape[0] != len(original_coords_np):
            raise ValueError("fault_features 的节点数与 original_coords 不匹配")

    # 构建数据字典
    try:
        data = {
            'node': nodes_np,
            'predicted_level': predicted_level_np[nodes_np].squeeze(),
            'predicted_rock_units': predicted_rock_units_np[nodes_np].squeeze(),
            'original_coords_x': original_coords_np[nodes_np, 0],
            'original_coords_y': original_coords_np[nodes_np, 1],
            'original_coords_z': original_coords_np[nodes_np, 2],
        }

        # 添加断层特征到数据字典
        num_faults = fault_features_np.shape[1]
        for i in range(num_faults):
            data[f'fault_feature_{i}'] = fault_features_np[nodes_np, i]

    except Exception as e:
        print("构建数据字典时出错:", e)
        print(f"nodes_np: {nodes_np}")
        print(f"predicted_level_np: {predicted_level_np}")
        print(f"predicted_rock_units: {predicted_rock_units_np}")
        print(f"original_coords_np: {original_coords_np}")
        print(f"fault_features_np: {fault_features_np}")
        raise e

    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    csv_filename = os.path.join(result_dir, f'rock_training_results{suffix}.csv')
    df.to_csv(csv_filename, index=False)
    print(f"预测结果已保存到 {csv_filename}")
