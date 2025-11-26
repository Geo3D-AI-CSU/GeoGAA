import numpy as np
import torch
from numpy.lib.function_base import gradient
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from itertools import combinations
import os
from maths import  azimuthplunge2vector,strikedip2vector


# 加载节点数据  ########这个的最新版本的图数据创建函数 有属性值的那个
def load_node_data_beifen(node_file,is_gradient=False):
    if is_gradient == False:
        with open(node_file, 'r') as f:
            header = f.readline()
            n_nodes = int(header.strip().split()[0])
        node_df = pd.read_csv(
            node_file,
            delim_whitespace=True,
            skiprows=1,
            nrows=n_nodes,
            header=None,
            engine='c',
            dtype={0: np.int32, 1: np.float32, 2: np.float32, 3: np.float32,
                   4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32, 8: str}  # 最后一列为字符串类型
        )
        node_data = node_df.values
    else:
        with open(node_file, 'r') as f:
            header = f.readline()
            n_nodes = int(header.strip().split()[0])
        node_df = pd.read_csv(
            node_file,
            delim_whitespace=True,
            skiprows=1,
            nrows=n_nodes,
            header=None,
            engine='c',
            dtype={0: np.int32, 1: np.float32, 2: np.float32, 3: np.float32,
                   4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32, 8: np.float32, 9: str,}  # 最后一列为字符串类型
        )
        node_data = node_df.values
    return node_data

def load_node_data(node_file):
    with open(node_file, 'r') as f:
        header = f.readline()
        n_nodes = int(header.strip().split()[0])
    node_df = pd.read_csv(
        node_file,
        delim_whitespace=True,
        skiprows=1,
        nrows=n_nodes,
        header=None,
        engine='c',
        dtype=np.float32
    )
    node_data = node_df.values
    return node_data


# 加载边数据  ########这个的最新版本的图数据创建函数 有属性值的那个
def load_edge_data_beifen(ele_file):
    with open(ele_file, 'r') as f:
        header = f.readline()
    edge_df = pd.read_csv(
        ele_file,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        usecols=[1, 2, 3, 4],
        engine='c',
        dtype=np.int32,
        on_bad_lines='skip'
    )
    edge_data = edge_df.values # 从1开始的索引转换为从0开始
    # edge_data = edge_df.values - 1  #0220之前的版本
    return edge_data
def load_edge_data(ele_file):
    with open(ele_file, 'r') as f:
        header = f.readline()
    edge_df = pd.read_csv(
        ele_file,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        usecols=[1, 2, 3, 4],
        engine='c',
        dtype=np.int32,
        on_bad_lines='skip'
    )
    edge_data = edge_df.values
    return edge_data


def create_graph(node_data, ele_data,is_gradient=False):
    if is_gradient == False:
        coords = node_data[:, 1:4]  # 提取 x, y, z
        properties = node_data[:, 4:]  # 其他属性，rock_unit, QJ,QX,level

        # 提取标签数据
        level = properties[:, -1]  # level (iso-value fv)
        QJ = properties[:, 1]  # QJ
        QX = properties[:,2]   # QX
        gradient = strikedip2vector(QJ, QX)
        dx = gradient[:,0].astype(np.float32)
        dy = gradient[:,1].astype(np.float32)
        dz = gradient[:,2].astype(np.float32)
        rock_unit = properties[:, 0]  # rock_unit (地质单元向量 yv)
        # 创建掩码
        mask_level = (~np.isnan(level)) & (level != -9999)
        mask_rock_unit = (~np.isnan(rock_unit)) & (rock_unit != -9999)
        mask_gradient = (~np.isnan(QJ)) & (QJ != -9999) & (~np.isnan(QX)) & (QX != -9999)

    else:
        coords = node_data[:, 1:4]  # 提取 x, y, z
        properties = node_data[:, 4:]  # 其他属性，rock_unit, dx, dy, dz, level

        # 提取标签数据
        level = properties[:, -1]  # level (iso-value fv)
        dx = properties[:, 1]  # dx
        dy = properties[:, 2]  # dy
        dz = properties[:, 3]  # dz
        rock_unit = properties[:, 0]  # rock_unit (地质单元向量 yv)
        # 创建掩码
        mask_level = (~np.isnan(level)) & (level != -9999)
        mask_rock_unit = (~np.isnan(rock_unit)) & (rock_unit != -9999)
        mask_gradient = (~np.isnan(dx)) & (dx != -9999) & (~np.isnan(dy)) & (dy != -9999) & (~np.isnan(dz)) & (dz != -9999)

    edge_index = []
    edge_set = set()
    for tetra in ele_data:
        for u, v in combinations(tetra, 2):
            # 标准化为有序元组实现去重
            sorted_edge = tuple(sorted((u, v)))
            edge_set.add(sorted_edge)

    # 转换为零基索引的 PyTorch 格式
    edge_index = torch.tensor(
        [[u - 1, v - 1] for u, v in edge_set],  # 假设原始节点编号从1开始
        dtype=torch.long
    ).t().contiguous()

    # 归一化坐标
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_coords = scaler.fit_transform(coords)
    np.save('normalization_params.npy', {'min': scaler.data_min_, 'range': scaler.data_range_})

    # 节点特征仅包含归一化的 (x, y, z)
    node_features = torch.tensor(normalized_coords, dtype=torch.float)

    # 创建图数据对象
    graph_data = Data(
        x=node_features,  # 节点特征 (x, y, z)
        edge_index=edge_index,
        mask_rock_unit=torch.tensor(mask_rock_unit, dtype=torch.bool),
        mask_level=torch.tensor(mask_level, dtype=torch.bool),
        mask_gradient=torch.tensor(mask_gradient, dtype=torch.bool),
        level=torch.tensor(level, dtype=torch.float),  # level 标签 (iso-value)
        rock_unit=torch.tensor(rock_unit, dtype=torch.long),  # rock_unit 标签 (地质单元)
        gradient=torch.tensor(np.stack((dx, dy, dz), axis=-1), dtype=torch.float),  # 法向量 (dx, dy, dz)
        original_coords=torch.tensor(coords, dtype=torch.float)  # 原始坐标
    )

    # 调试信息
    print("Graph Data Attributes after creation:", list(graph_data.keys))
    print("Level Attribute Shape:", graph_data.level.shape)
    print("Sample Level Data:", graph_data.level[:5])

    return graph_data


def create_graph_old_edge(node_data, ele_data,is_gradient=False):
    if is_gradient == False:
        coords = node_data[:, 1:4]  # 提取 x, y, z
        properties = node_data[:, 4:]  # 其他属性，rock_unit, QJ,QX,level

        # 提取标签数据
        level = properties[:, -1]  # level (iso-value fv)
        QJ = properties[:, 1]  # QJ
        QX = properties[:,2]   # QX
        gradient = strikedip2vector(QJ, QX)
        dx = gradient[:,0].astype(np.float32)
        dy = gradient[:,1].astype(np.float32)
        dz = gradient[:,2].astype(np.float32)
        rock_unit = properties[:, 0]  # rock_unit (地质单元向量 yv)

    else:
        coords = node_data[:, 1:4]  # 提取 x, y, z
        properties = node_data[:, 4:]  # 其他属性，rock_unit, dx, dy, dz, level

        # 提取标签数据
        level = properties[:, -1]  # level (iso-value fv)
        dx = properties[:, 1]  # dx
        dy = properties[:, 2]  # dy
        dz = properties[:, 3]  # dz
        rock_unit = properties[:, 0]  # rock_unit (地质单元向量 yv)

    # 创建掩码
    mask_level = (~np.isnan(level)) & (level != -9999)
    mask_rock_unit = (~np.isnan(rock_unit)) & (rock_unit != -9999)
    mask_gradient = (~np.isnan(dx)) & (dx != -9999) & (~np.isnan(dy)) & (dy != -9999) & (~np.isnan(dz)) & (dz != -9999)

    # # 创建边索引 (邻接矩阵)   删除重复的边----------------
    edge_list = []
    for tetra in ele_data:
        # 使用 combinations 获取所有的边，组合每对节点 (u, v)
        edges = list(combinations(tetra, 2))  # [(n1, n2), (n1, n3), ...]
        # 排序节点索引，确保 (min, max)
        sorted_edges = [tuple(sorted(edge)) for edge in edges]
        edge_list.extend(sorted_edges)
    # 转换为numpy数组
    edge_array = np.array(edge_list, dtype=np.int32)
    # 去除重复的边
    edge_unique = np.unique(edge_array, axis=0)
    # 转置为 (2, num_edges) 并转换为torch tensor
    edge_index = torch.tensor(edge_unique.T, dtype=torch.long)

    # 归一化坐标
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_coords = scaler.fit_transform(coords)
    np.save('normalization_params.npy', {'min': scaler.data_min_, 'range': scaler.data_range_})

    # 节点特征仅包含归一化的 (x, y, z)
    node_features = torch.tensor(normalized_coords, dtype=torch.float)

    # 创建图数据对象
    graph_data = Data(
        x=node_features,  # 节点特征 (x, y, z)
        edge_index=edge_index,
        mask_rock_unit=torch.tensor(mask_rock_unit, dtype=torch.bool),
        mask_level=torch.tensor(mask_level, dtype=torch.bool),
        mask_gradient=torch.tensor(mask_gradient, dtype=torch.bool),
        level=torch.tensor(level, dtype=torch.float),  # level 标签 (iso-value)
        rock_unit=torch.tensor(rock_unit, dtype=torch.long),  # rock_unit 标签 (地质单元)
        gradient=torch.tensor(np.stack((dx, dy, dz), axis=-1), dtype=torch.float),  # 法向量 (dx, dy, dz)
        original_coords=torch.tensor(coords, dtype=torch.float)  # 原始坐标
    )

    # 调试信息
    print("Graph Data Attributes after creation:", list(graph_data.keys))
    print("Level Attribute Shape:", graph_data.level.shape)
    print("Sample Level Data:", graph_data.level[:5])

    return graph_data

########这个的最新版本的图数据创建函数 有属性值的那个
def create_graph_beifen(node_data, ele_data,is_gradient=False):
    if is_gradient == False:
        coords = node_data[:, 1:4].astype(np.float32)  # 提取 x, y, z
        properties = node_data[:, 4:-1].astype(np.float32)  # 其他属性，Level,Rock_unit,QJ,QX,Attribute

        # 提取标签数据
        attribute = node_data[:, -1]  #直接从节点读取保持dtype=object
        rock_unit = properties[:, 1]
        QJ = properties[:, 2]
        QX = properties[:, 3]
        level= properties[:, 0]
        gradient = strikedip2vector(QJ, QX)
        dx =  gradient[:, 0].astype(np.float32)
        dy =  gradient[:, 1].astype(np.float32)
        dz =  gradient[:, 2].astype(np.float32)
    else:
        coords = node_data[:, 1:4].astype(np.float32)  # 提取 x, y, z
        properties = node_data[:, 4:-1].astype(np.float32)  # 其他属性，Level,Rock_unit,dx,dy,dz,Attribute
        # 提取标签数据
        attribute = node_data[:, -1]  # 直接从节点读取保持dtype=object
        rock_unit = properties[:, 1]
        level = properties[:, 0]
        dx = properties[:, 2]
        dy = properties[:, 3]
        dz = properties[:, 4]
    # 创建掩码
    mask_level = (level != -9999)
    mask_rock_unit = (rock_unit != -9999)
    mask_gradient = (~np.isnan(dx)) & (dx != -9999) & (~np.isnan(dy)) & (dy != -9999) & (~np.isnan(dz)) & (dz != -9999)


    # # 创建边索引 (邻接矩阵)   删除重复的边----------------
    edge_list = []
    for tetra in ele_data:
        # 使用 combinations 获取所有的边，组合每对节点 (u, v)
        edges = list(combinations(tetra, 2))  # [(n1, n2), (n1, n3), ...]
        # 排序节点索引，确保 (min, max)
        sorted_edges = [tuple(sorted(edge)) for edge in edges]
        edge_list.extend(sorted_edges)
    # 转换为numpy数组
    edge_array = np.array(edge_list, dtype=np.int32)
    # 去除重复的边
    edge_unique = np.unique(edge_array, axis=0)
    # 转置为 (2, num_edges) 并转换为torch tensor
    edge_index = torch.tensor(edge_unique.T, dtype=torch.long)




    # 归一化坐标
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_coords = scaler.fit_transform(coords)
    # np.save('normalization_params.npy', {'min': scaler.data_min_, 'range': scaler.data_range_})

    # 节点特征仅包含归一化的 (x, y, z)
    node_features = torch.tensor(normalized_coords, dtype=torch.float)

    # 创建图数据对象
    graph_data = Data(
        x=node_features,  # 节点特征 (x, y, z)
        edge_index=edge_index,
        mask_rock_unit=torch.tensor(mask_rock_unit, dtype=torch.bool),
        mask_level=torch.tensor(mask_level, dtype=torch.bool),
        mask_gradient=torch.tensor(mask_gradient, dtype=torch.bool),
        level=torch.tensor(level, dtype=torch.float),  # level 标签 (iso-value)
        rock_unit=torch.tensor(rock_unit, dtype=torch.long),  # rock_unit 标签 (地质单元)
        gradient=torch.tensor(np.stack((dx, dy, dz), axis=-1), dtype=torch.float),  # 法向量 (dx, dy, dz)
        original_coords=torch.tensor(coords, dtype=torch.float),  # 原始坐标
        attribute=attribute
    )

    # 调试信息
    print("Graph Data Attributes after creation:", list(graph_data.keys))
    print("Level Attribute Shape:", graph_data.level.shape)
    print("Sample Level Data:", graph_data.level[:5])

    return graph_data


# 加载或创建图数据
def create_or_load_graph(node_file, ele_file, pt_file=None,is_gradient=False):
    """
    检查是否存在已保存的图数据pt文件。如果存在，直接加载，否则创建图结构数据并保存。
    """
    if pt_file is None:
        node_dir = os.path.dirname(node_file)
        pt_file = os.path.join(node_dir, 'graph_data.pt')
    if os.path.exists(pt_file):
        print(f"加载已存在的图数据：{pt_file}")
        graph_data = torch.load(pt_file)
    else:
        print("未发现已保存的图数据，正在创建...")
        node_data = load_node_data(node_file)
        edge_data = load_edge_data(ele_file)
        graph_data = create_graph(node_data, edge_data,is_gradient=is_gradient)
        torch.save(graph_data, pt_file)
        print(f"图数据已保存为：{pt_file}")

    return graph_data