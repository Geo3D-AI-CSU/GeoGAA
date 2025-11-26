from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import subgraph
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 方案1: 基础三层扩展版本
# ============================================================================
class GATSAGEMultiTaskPredictor_V1(nn.Module):
    """
    改进版本1: 基础三层扩展

    架构说明:
    - GAT: 3层图注意力网络，捕获局部注意力模式
    - GraphSAGE: 3层邻域聚合网络，聚合邻域信息
    - 特征融合: 简单拼接 + 线性变换
    - 任务头: 分别用于Level预测(回归)和Rock分类(分类)

    改进点:
    1. 将GAT和GraphSAGE都扩展到3层
    2. 每层后添加LayerNorm稳定训练
    3. GAT最后一层使用平均聚合(concat=False)
    4. 适当的Dropout防止过拟合

    参数:
    - in_channels: 输入特征维度 (3 + 断层特征数)
    - hidden_channels: 隐藏层维度
    - gat_heads: GAT注意力头数
    - num_classes: 岩性分类类别数
    - dropout: Dropout概率
    - activation_fn: 激活函数类型 ('prelu' or 'softplus')
    """

    def __init__(self, in_channels, hidden_channels=128, gat_heads=2,
                 num_classes=13, dropout=0.0, activation_fn='prelu'):
        super(GATSAGEMultiTaskPredictor_V1, self).__init__()

        self.dropout = dropout
        self.gat_heads = gat_heads

        print(f"\n{'=' * 80}")
        print(f"🏗️  初始化 ")
        print(f"{'=' * 80}")
        print(f"  输入维度: {in_channels}")
        print(f"  隐藏维度: {hidden_channels}")
        print(f"  GAT注意力头数: {gat_heads}")
        print(f"  岩性类别数: {num_classes}")
        print(f"  Dropout概率: {dropout}")
        print(f"  激活函数: {activation_fn}")
        print(f"{'=' * 80}\n")

        # ========== GAT特征提取 (3层) ==========
        # 第1层: in_channels -> hidden_channels * gat_heads
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=gat_heads,
            dropout=dropout,
            concat=True  # 拼接多头输出
        )
        self.gat_norm1 = nn.LayerNorm(hidden_channels * gat_heads)

        # 第2层: hidden_channels * gat_heads -> hidden_channels * gat_heads
        self.gat2 = GATConv(
            hidden_channels * gat_heads,
            hidden_channels,
            heads=gat_heads,
            dropout=dropout,
            concat=True
        )
        self.gat_norm2 = nn.LayerNorm(hidden_channels * gat_heads)

        # 第3层: hidden_channels * gat_heads -> hidden_channels
        # 注意: concat=False 表示对多头输出取平均而非拼接
        self.gat3 = GATConv(
            hidden_channels * gat_heads,
            hidden_channels,
            heads=gat_heads,
            dropout=dropout,
            concat=False  # 平均聚合，输出维度为 hidden_channels
        )
        self.gat_norm3 = nn.LayerNorm(hidden_channels)

        # ========== GraphSAGE特征提取 (3层) ==========
        # 接收GAT的输出，维度为 hidden_channels
        self.sage1 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.sage_norm1 = nn.LayerNorm(hidden_channels)

        self.sage2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.sage_norm2 = nn.LayerNorm(hidden_channels)

        self.sage3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.sage_norm3 = nn.LayerNorm(hidden_channels)

        # ========== 特征融合层 ==========
        # 将GAT和SAGE的输出拼接后融合
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_norm = nn.LayerNorm(hidden_channels)

        # ========== MLP任务头 ==========
        # Level预测分支 (回归任务)
        self.level_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1)
        )

        # 岩性分类分支 (分类任务)
        self.rock_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, num_classes)
        )

        # 激活函数
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"{'=' * 80}\n")

    def forward(self, x, edge_index):
        """
        前向传播

        参数:
        - x: 节点特征 [num_nodes, in_channels]
        - edge_index: 边索引 [2, num_edges]

        返回:
        - level_output: Level预测值 [num_nodes]
        - rock_output: 岩性分类logits [num_nodes, num_classes]
        """

        # ========== GAT特征提取 (3层) ==========
        # Layer 1
        gat_out = self.gat1(x, edge_index)
        gat_out = self.gat_norm1(gat_out)
        gat_out = self.activation(gat_out)
        gat_out = F.dropout(gat_out, p=self.dropout, training=self.training)

        # Layer 2
        gat_out = self.gat2(gat_out, edge_index)
        gat_out = self.gat_norm2(gat_out)
        gat_out = self.activation(gat_out)
        gat_out = F.dropout(gat_out, p=self.dropout, training=self.training)

        # Layer 3 (平均聚合)
        gat_out = self.gat3(gat_out, edge_index)
        gat_out = self.gat_norm3(gat_out)
        gat_out = self.activation(gat_out)
        gat_out = F.dropout(gat_out, p=self.dropout, training=self.training)

        # ========== GraphSAGE特征提取 (3层) ==========
        # Layer 1
        sage_out = self.sage1(gat_out, edge_index)
        sage_out = self.sage_norm1(sage_out)
        sage_out = self.activation(sage_out)
        sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)

        # Layer 2
        sage_out = self.sage2(sage_out, edge_index)
        sage_out = self.sage_norm2(sage_out)
        sage_out = self.activation(sage_out)
        sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)

        # Layer 3
        sage_out = self.sage3(sage_out, edge_index)
        sage_out = self.sage_norm3(sage_out)
        sage_out = self.activation(sage_out)
        sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)

        # ========== 特征融合 ==========
        # 拼接GAT和SAGE的输出
        fused_features = torch.cat([gat_out, sage_out], dim=-1)
        fused_features = self.fusion(fused_features)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.activation(fused_features)
        fused_features = F.dropout(fused_features, p=self.dropout, training=self.training)

        # ========== 多任务预测 ==========
        # Level预测 (回归)
        level_output = self.level_mlp(fused_features).squeeze(-1)

        # 岩性分类
        rock_output = self.rock_mlp(fused_features)

        return level_output, rock_output


# ============================================================================
# 保留原有模型以保持兼容性
# ============================================================================
class GATSAGEMultiTaskPredictor(nn.Module):
    """
    原始版本 (兼容性保留)
    GAT: 2层 + GraphSAGE: 2层
    """

    def __init__(self, in_channels, hidden_channels=128, gat_heads=2,
                 num_classes=13, dropout=0.0, activation_fn='prelu'):
        super(GATSAGEMultiTaskPredictor, self).__init__()

        self.dropout = dropout
        self.gat_heads = gat_heads

        # ========== 图特征提取层 ==========
        # GAT层 - 捕获局部注意力模式
        self.gat1 = GATConv(in_channels, hidden_channels, heads=gat_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * gat_heads, hidden_channels, heads=gat_heads, dropout=dropout)

        # GraphSAGE层 - 聚合邻域信息
        self.sage1 = SAGEConv(hidden_channels * gat_heads, hidden_channels, aggr='mean')
        self.sage2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

        # 特征融合层 - 融合GAT和SAGE的输出
        self.fusion = nn.Linear(hidden_channels * gat_heads + hidden_channels, hidden_channels)

        # ========== MLP任务头 ==========
        # Level预测分支 (回归)
        self.level_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1)
        )

        # 岩性分类分支 (分类)
        self.rock_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.PReLU() if activation_fn == 'prelu' else nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, num_classes)
        )

        # 激活函数
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index):
        # ========== GAT特征提取 ==========
        gat_out = self.gat1(x, edge_index)
        gat_out = self.activation(gat_out)
        gat_out = F.dropout(gat_out, p=self.dropout, training=self.training)

        gat_out = self.gat2(gat_out, edge_index)
        gat_out = self.activation(gat_out)
        gat_out = F.dropout(gat_out, p=self.dropout, training=self.training)

        # ========== GraphSAGE特征提取 ==========
        sage_out = self.sage1(gat_out, edge_index)
        sage_out = self.activation(sage_out)
        sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)

        sage_out = self.sage2(sage_out, edge_index)
        sage_out = self.activation(sage_out)
        sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)

        # ========== 特征融合 ==========
        fused_features = torch.cat([gat_out, sage_out], dim=-1)
        fused_features = self.fusion(fused_features)
        fused_features = self.activation(fused_features)
        fused_features = F.dropout(fused_features, p=self.dropout, training=self.training)

        # ========== 多任务预测 ==========
        level_output = self.level_mlp(fused_features).squeeze(-1)
        rock_output = self.rock_mlp(fused_features)

        return level_output, rock_output


# ============================================================================
# 其他保留的模型类 (保持兼容性)
# ============================================================================

class GATLevelPredictor(nn.Module):
    """GAT Level预测器 (单任务)"""

    def __init__(self, in_channels, hidden_channels, heads=2, dropout=0, activation_fn='prelu'):
        super(GATLevelPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(
            self.embed_dim * heads,
            self.embed_dim,
            heads=heads,
            dropout=dropout,
            concat=False
        )

        self.level_predictor = nn.Linear(self.embed_dim, 1)

        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        level_output = self.level_predictor(x).squeeze(-1)
        return level_output


class LevelPredictor(torch.nn.Module):
    """GraphSAGE Level预测器 (单任务)"""

    def __init__(self, in_channels, hidden_channels=128, out_channels=64,
                 activation_fn='prelu', dropout=0.0):
        super(LevelPredictor, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')

        self.level_predictor = torch.nn.Linear(out_channels, 1)

        if activation_fn == 'softplus':
            self.activation = torch.nn.Softplus()
        else:
            self.activation = torch.nn.PReLU()

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        if self.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        if self.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        level_output = self.level_predictor(x).squeeze(-1)

        return level_output


class RockUnitPredictor(torch.nn.Module):
    """GraphSAGE 岩性预测器 (单任务)"""

    def __init__(self, in_channels, hidden_channels=128, out_channels=64,
                 num_classes=4, dropout=0.0):
        super(RockUnitPredictor, self).__init__()

        self.conv1 = SAGEConv(in_channels + 1, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.rock_unit_classifier = torch.nn.Linear(out_channels, num_classes)
        self.prelu = torch.nn.PReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        x = self.conv3(x, edge_index)

        rock_unit_output = self.rock_unit_classifier(x)
        return rock_unit_output


class GATRockPredictor(nn.Module):
    """GAT 岩性预测器 (单任务)"""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 heads=2, dropout=0, activation_fn='prelu'):
        super(GATRockPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(
            self.embed_dim * heads,
            self.embed_dim,
            heads=heads,
            dropout=dropout,
            concat=False
        )

        self.rock_classifier = nn.Linear(self.embed_dim, num_classes)

        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        rock_unit_output = self.rock_classifier(x)
        return rock_unit_output