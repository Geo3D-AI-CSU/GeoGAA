import torch
import numpy as np
if not hasattr(np,'bool'):
    np.bool = bool
import pyvista as pv
import os
import re
import glob
import vtk


def read_vtk_files(vtk_directory, pattern='F*.vtk'):
    """
    读取指定目录下所有符合模式的 .vtk 文件，并提取 level 值。

    参数:
    - vtk_directory (str): 存放 .vtk 文件的目录路径。
    - pattern (str): 文件名匹配模式，默认 'F*.vtk'。

    返回:
    - fault_meshes (dict): 以 level 为键，PyVista.PolyData 对象为值的字典。
    """
    vtk_files = glob.glob(os.path.join(vtk_directory, pattern))
    fault_meshes = {}
    for vtk_file in vtk_files:
        # 从文件名中提取 level，例如 'F37.vtk' 中提取 37
        match = re.search(r'F(\d+)\.vtk', os.path.basename(vtk_file))
        if match:
            level = int(match.group(1))
            mesh = pv.read(vtk_file)
            fault_meshes[level] = mesh
            print(f"已读取断层面 Level {level} 的文件: {vtk_file}")
        else:
            print(f"文件名 {vtk_file} 不符合 'F{{level}}.vtk' 格式，已跳过。")
    return fault_meshes

def compute_fault_features(graph_data, vtk_directory,factor=1.0):
    """
    生成断层特征，计算节点与断层面的关系，并分配断层侧别特征。

    参数:
    - graph_data (torch_geometric.data.Data): 图数据对象，必须包含 'original_coords'。
    - vtk_directory (str): 存放 .vtk 文件的目录路径。
    - visualize (bool): 是否可视化断层面。
    - output_dir (str): 保存可视化图像的目录。

    返回:
    - graph_data (torch_geometric.data.Data): 更新后的图数据对象，包含断层特征。
    """
    # 提取节点坐标
    coords = graph_data.original_coords.cpu().numpy()
    num_nodes = coords.shape[0]

    # 读取所有断层面的 vtk 文件
    fault_meshes = read_vtk_files(vtk_directory)

    num_levels = len(fault_meshes)
    fault_features = np.zeros((num_nodes, num_levels), dtype=int)

    # 创建全局边界盒子（可选，根据需要调整）
    x_min, y_min, z_min = coords.min(axis=0) - 10.0  # 添加缓冲区
    x_max, y_max, z_max = coords.max(axis=0) + 10.0
    global_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    print(f"全局边界: {global_bounds}")

    # 遍历每个断层面，计算有符号距离并分配断层特征
    for idx, (level, mesh) in enumerate(sorted(fault_meshes.items())):
        # 获取法向量数据
        normals = mesh.GetPointData().GetNormals()
        try:
            if normals is None:
                mesh.compute_normals(inplace=True)

            # 将图数据的坐标转换为 VTK 格式
            vtk_points = vtk.vtkPoints()
            for coord in coords:  # coords 应该是图数据中的节点坐标
                vtk_points.InsertNextPoint(coord)

            # 创建 VTK PolyData 对象，并将节点坐标赋给它
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(vtk_points)

            # 将 VTK PolyData 转换为 PyVista PolyData
            pyvista_poly_data = pv.wrap(poly_data)

            # 使用 PyVista 包装断层面 mesh
            pyvista_mesh = pv.wrap(mesh)

            # 计算节点到断层面的距离，正确的参数顺序是：断层面作为第一个参数，点集作为第二个参数
            result = pyvista_poly_data.compute_implicit_distance(pyvista_mesh,inplace=True)
            distances = result['implicit_distance']
            # if level == 2:
            #     distances = -distances
            # if level == 38:
            #     distances = -distances
            if level == 1:
                distances = -distances   #看下是否跟论文的编码策略一致
            # 根据距离的正负来判断节点的上盘下盘
            fault_sides = (distances > 0).astype(int)  # 距离 > 0 为 1（上盘），<= 0 为 0（下盘）
            # 将结果存储到 fault_features 中
            fault_features[:, idx] = fault_sides
            # 输出每个断层面 Level 的处理信息
            print(f"断层面 Level {level} 的节点已分配到 {'上盘' if (fault_sides.sum() / num_nodes) > 0.5 else '下盘'}。")
        except Exception as e:
            print(f"处理断层面 Level {level} 时出现错误: {e}")

    # 将断层特征转换为张量并移动到设备
    fault_features_tensor = torch.tensor(fault_features, dtype=torch.float32).to(graph_data.x.device)
    # -1到1
    # fault_features_tensor = 2*fault_features_tensor - 1
    # fault_features_tensor =  fault_features_tensor*factor
    # 将断层特征拼接到现有的节点特征中
    graph_data.x = torch.cat([graph_data.x, fault_features_tensor], dim=-1)
    print("断层特征已添加到 graph_data.x。")

    return graph_data