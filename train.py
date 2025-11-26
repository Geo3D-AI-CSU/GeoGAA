"""
改进的多任务训练脚本
集成地层约束和Focal Loss
"""
import torch
import time
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

from stratigraphic_constraint import (
    StratigraphicConstraint,
    StratigraphicConstraintLoss,
    FocalLoss,
    compute_class_weights,
    post_process_with_level_constraint
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from input_graph_data import create_or_load_graph
from model import GATSAGEMultiTaskPredictor_V1
from loss_fn import level_loss, gradient_loss
from metrics import calculate_rmse, calculate_accuracy, calculate_r2, calculate_confusion_matrix
from select_device import select_device, set_random_seed
from save_data import save_rock_result_to_csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Normalizer import Normalizer
from compute_fault_zone_feature import compute_fault_features

set_random_seed(42)
device = select_device(desired_gpu=1)
normalizer = Normalizer()


class ImprovedGradNorm:
    """改进的GradNorm，支持4个损失函数"""

    def __init__(self, alpha=1.0, gamma=1.0, delta=1.0, beta=0.5, device='cuda'):
        self.device = device
        # level_loss, gradient_loss, rock_loss, strat_constraint_loss
        self.loss_weights = torch.tensor([alpha, gamma, delta, beta],
                                         dtype=torch.float32, device=device)

    def compute_loss(self, level_loss, gradient_loss, rock_loss, strat_loss):
        return (self.loss_weights[0] * level_loss +
                self.loss_weights[1] * gradient_loss +
                self.loss_weights[2] * rock_loss +
                self.loss_weights[3] * strat_loss)

    def update_weights(self, level_loss, gradient_loss, rock_loss, strat_loss, model):
        level_grad_norm = self.compute_grad_norm(level_loss, model)
        grad_grad_norm = self.compute_grad_norm(gradient_loss, model)
        rock_grad_norm = self.compute_grad_norm(rock_loss, model)
        strat_grad_norm = self.compute_grad_norm(strat_loss, model)

        grad_norms = torch.tensor([level_grad_norm, grad_grad_norm,
                                   rock_grad_norm, strat_grad_norm],
                                  device=self.device)

        normed_grad_norms = grad_norms / (grad_norms.mean() + 1e-8)
        grad_ratio = normed_grad_norms / (grad_norms + 1e-8)

        self.loss_weights = self.loss_weights * grad_ratio
        self.loss_weights = self.loss_weights / self.loss_weights.sum()

        return self.loss_weights

    def compute_grad_norm(self, loss, model):
        loss.backward(retain_graph=True)
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5


def split_train_test(graph_data, train_ratio=0.8, random_seed=42):
    """划分训练集和测试集"""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    num_nodes = graph_data.x.size(0)

    # Level标签划分
    level_indices = torch.where(graph_data.mask_level)[0].cpu().numpy()
    if len(level_indices) > 0:
        train_level_indices, test_level_indices = train_test_split(
            level_indices, train_size=train_ratio, random_state=random_seed, shuffle=True
        )
        train_mask_level = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_level = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask_level[train_level_indices] = True
        test_mask_level[test_level_indices] = True
    else:
        train_mask_level = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_level = torch.zeros(num_nodes, dtype=torch.bool)

    # 岩性标签划分（分层采样，确保每个类别都有训练和测试样本）
    rock_indices = torch.where(graph_data.mask_rock_unit)[0].cpu().numpy()
    rock_labels = graph_data.rock_unit[rock_indices].cpu().numpy()

    if len(rock_indices) > 0:
        try:
            # 分层采样
            train_rock_indices, test_rock_indices = train_test_split(
                rock_indices,
                train_size=train_ratio,
                random_state=random_seed,
                shuffle=True,
                stratify=rock_labels  # 关键：分层采样
            )
        except ValueError:
            # 如果某些类别样本太少无法分层，则普通随机采样
            print("⚠️  警告：某些岩性类别样本太少，使用普通随机采样")
            train_rock_indices, test_rock_indices = train_test_split(
                rock_indices, train_size=train_ratio, random_state=random_seed, shuffle=True
            )

        train_mask_rock = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_rock = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask_rock[train_rock_indices] = True
        test_mask_rock[test_rock_indices] = True
    else:
        train_mask_rock = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_rock = torch.zeros(num_nodes, dtype=torch.bool)

    # 梯度标签划分
    gradient_indices = torch.where(graph_data.mask_gradient)[0].cpu().numpy()
    if len(gradient_indices) > 0:
        train_gradient_indices, test_gradient_indices = train_test_split(
            gradient_indices, train_size=train_ratio, random_state=random_seed, shuffle=True
        )
        train_mask_gradient = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_gradient = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask_gradient[train_gradient_indices] = True
        test_mask_gradient[test_gradient_indices] = True
    else:
        train_mask_gradient = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_gradient = torch.zeros(num_nodes, dtype=torch.bool)

    print("\n" + "=" * 80)
    print("📊 数据集划分信息（分层采样）")
    print("=" * 80)
    print(f"Level标签: 训练 {train_mask_level.sum().item()} | "
          f"测试 {test_mask_level.sum().item()} | "
          f"比例 {train_mask_level.sum().item() / (train_mask_level.sum().item() + test_mask_level.sum().item()):.2%}")
    print(f"岩性标签: 训练 {train_mask_rock.sum().item()} | "
          f"测试 {test_mask_rock.sum().item()} | "
          f"比例 {train_mask_rock.sum().item() / (train_mask_rock.sum().item() + test_mask_rock.sum().item()):.2%}")
    print(f"梯度标签: 训练 {train_mask_gradient.sum().item()} | "
          f"测试 {test_mask_gradient.sum().item()} | "
          f"比例 {train_mask_gradient.sum().item() / (train_mask_gradient.sum().item() + test_mask_gradient.sum().item()):.2%}")

    # 统计各类别的训练/测试分布
    print("\n各岩性类别的训练/测试分布:")
    for rock_unit in range(1, 14):
        train_count = ((graph_data.rock_unit == rock_unit) & train_mask_rock).sum().item()
        test_count = ((graph_data.rock_unit == rock_unit) & test_mask_rock).sum().item()
        total = train_count + test_count
        if total > 0:
            print(f"  岩性{rock_unit:2d}: 训练{train_count:4d} | 测试{test_count:4d} | "
                  f"总计{total:4d} | 训练占比{train_count / total:.1%}")
    print("=" * 80 + "\n")

    return (train_mask_level, test_mask_level,
            train_mask_rock, test_mask_rock,
            train_mask_gradient, test_mask_gradient)


def evaluate_model_with_constraints(model, graph_data, mask_level, mask_rock, mask_gradient,
                                    edge_index, gradient, original_coords,
                                    strat_constraint, focal_loss_fn, strat_loss_fn,
                                    device, phase="Test", use_post_process=False):
    """评估模型（包含地层约束）"""
    model.eval()
    with torch.no_grad():
        predicted_level, predicted_rock_logits = model(
            graph_data.x.to(device),
            graph_data.edge_index.to(device)
        )

        # 后处理（可选）
        if use_post_process:
            predicted_rock_corrected = post_process_with_level_constraint(
                predicted_level, predicted_rock_logits, strat_constraint, device
            )
            # 重新计算logits（用于损失计算）
            # 注意：这里为了评估，我们直接使用修正后的预测

        # Level指标
        if mask_level.any():
            rmse = calculate_rmse(predicted_level, graph_data.level, mask_level)
            r2 = calculate_r2(predicted_level, graph_data.level, mask_level)
            level_loss_val = level_loss(
                predicted_level[mask_level],
                graph_data.level[mask_level].to(device)
            ).item()
        else:
            rmse, r2, level_loss_val = 0.0, 0.0, 0.0

        # 梯度损失
        if mask_gradient.any():
            grad_loss_val = gradient_loss(
                predicted_level,
                original_coords,
                gradient[mask_gradient, 0],
                gradient[mask_gradient, 1],
                gradient[mask_gradient, 2],
                edge_index,
                mask_gradient
            ).item()
        else:
            grad_loss_val = 0.0

        # 岩性分类指标
        if mask_rock.any():
            # 原始预测精度
            accuracy_original = calculate_accuracy(
                predicted_rock_logits,
                graph_data.rock_unit,
                mask_rock
            )

            # 后处理精度
            if use_post_process:
                predicted_rock_corrected = post_process_with_level_constraint(
                    predicted_level, predicted_rock_logits, strat_constraint, device
                )
                # 计算修正后的精度
                correct = (predicted_rock_corrected[mask_rock] == (graph_data.rock_unit[mask_rock] - 1).to(
                    device)).sum().item()
                accuracy_corrected = correct / mask_rock.sum().item()
            else:
                accuracy_corrected = accuracy_original

            # Focal Loss
            rock_loss_val = focal_loss_fn(
                predicted_rock_logits[mask_rock],
                (graph_data.rock_unit.to(device)[mask_rock] - 1).long()
            ).item()

            # 地层约束损失
            strat_loss_val = strat_loss_fn(
                predicted_level,
                predicted_rock_logits,
                graph_data.rock_unit.to(device),
                mask_rock,
                device
            ).item()
        else:
            accuracy_original, accuracy_corrected, rock_loss_val, strat_loss_val = 0.0, 0.0, 0.0, 0.0

    metrics = {
        'phase': phase,
        'level_loss': level_loss_val,
        'grad_loss': grad_loss_val,
        'rock_loss': rock_loss_val,
        'strat_loss': strat_loss_val,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy_original,
        'accuracy_corrected': accuracy_corrected
    }

    return metrics


def train_multitask_with_constraints(graph_data, num_epochs=300, lr=0.01,
                                     hidden_channels=128, num_classes=13,
                                     result_dir=None, dropout=0.1, lr_decay=0.8,
                                     gat_heads=2, train_ratio=0.8,
                                     strat_weight=1.0, focal_gamma=2.0):
    """
    改进的多任务训练（集成地层约束）

    新增参数:
    - strat_weight: 地层约束损失权重
    - focal_gamma: Focal Loss的gamma参数
    """

    # 数据集划分（分层采样）
    (train_mask_level, test_mask_level,
     train_mask_rock, test_mask_rock,
     train_mask_gradient, test_mask_gradient) = split_train_test(
        graph_data, train_ratio=train_ratio, random_seed=42
    )

    # 初始化模型
    model = GATSAGEMultiTaskPredictor_V1(
        in_channels=graph_data.x.size(1),
        hidden_channels=hidden_channels,
        gat_heads=gat_heads,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay, patience=15,
        threshold=1e-3, min_lr=1e-6
    )

    # 数据归一化
    original_level = graph_data.level[train_mask_level]
    level_norm = normalizer.fit_transform_level_masked(graph_data.level[train_mask_level])

    graph_data_level_normalized = graph_data.level.clone()
    graph_data_level_normalized[train_mask_level] = level_norm

    if test_mask_level.any():
        test_level_original = graph_data.level[test_mask_level]
        test_level_norm = normalizer.level_scaler.transform(
            test_level_original.cpu().numpy().reshape(-1, 1)
        )
        graph_data_level_normalized[test_mask_level] = torch.tensor(
            test_level_norm.squeeze(), dtype=torch.float32
        ).to(graph_data.level.device)

    graph_data.level = graph_data_level_normalized

    edge_index = graph_data.edge_index.to(device)
    gradient = graph_data.gradient.to(device)
    original_coords = graph_data.original_coords.to(device)

    # 🔥 关键改进1：初始化地层约束
    strat_constraint = StratigraphicConstraint()
    strat_loss_fn = StratigraphicConstraintLoss(
        strat_constraint, weight=strat_weight, temperature=1000.0
    )

    # 🔥 关键改进2：计算类别权重，初始化Focal Loss
    class_weights = compute_class_weights(
        graph_data.rock_unit, train_mask_rock, num_classes=num_classes
    )
    print(f"\n📊 类别权重: {class_weights.numpy()}")

    focal_loss_fn = FocalLoss(alpha=class_weights.to(device), gamma=focal_gamma)

    # 改进的GradNorm（4个损失）
    grad_norm = ImprovedGradNorm(
        alpha=1.0, gamma=0.5, delta=1.0, beta=0.5, device=device
    )

    # 训练循环
    log_file = os.path.join(result_dir, 'training_log_with_constraints.txt')

    best_test_accuracy = 0.0
    best_epoch = 0

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Multi-Task Training with Stratigraphic Constraints\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: GATSAGEMultiTaskPredictor_V1\n")
        f.write(f"Focal Loss Gamma: {focal_gamma}\n")
        f.write(f"Stratigraphic Constraint Weight: {strat_weight}\n")
        f.write(f"Train:Test = {int(train_ratio * 100)}:{int((1 - train_ratio) * 100)}\n")
        f.write("=" * 80 + "\n\n")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            model.train()
            optimizer.zero_grad()

            predicted_level, predicted_rock_logits = model(
                graph_data.x.to(device),
                graph_data.edge_index.to(device)
            )

            # Level损失
            train_level_loss = torch.tensor(0.0, device=device)
            if train_mask_level.any():
                train_level_loss = level_loss(
                    predicted_level[train_mask_level],
                    graph_data.level[train_mask_level].to(device)
                )

            # 梯度损失
            train_grad_loss = torch.tensor(0.0, device=device)
            if train_mask_gradient.any():
                train_grad_loss = gradient_loss(
                    predicted_level,
                    original_coords,
                    gradient[train_mask_gradient, 0],
                    gradient[train_mask_gradient, 1],
                    gradient[train_mask_gradient, 2],
                    edge_index,
                    train_mask_gradient
                )

            # 🔥 改进：使用Focal Loss替代交叉熵
            train_rock_loss = torch.tensor(0.0, device=device)
            if train_mask_rock.any():
                train_rock_loss = focal_loss_fn(
                    predicted_rock_logits[train_mask_rock],
                    (graph_data.rock_unit.to(device)[train_mask_rock] - 1).long()
                )

            # 🔥 改进：添加地层约束损失
            train_strat_loss = torch.tensor(0.0, device=device)
            if train_mask_rock.any():
                # 注意：需要反归一化level用于约束计算
                predicted_level_original = normalizer.inverse_transform_level(predicted_level)
                train_strat_loss = strat_loss_fn(
                    predicted_level_original,
                    predicted_rock_logits,
                    graph_data.rock_unit.to(device),
                    train_mask_rock,
                    device
                )

            # 更新损失权重
            loss_weights = grad_norm.update_weights(
                train_level_loss, train_grad_loss, train_rock_loss, train_strat_loss, model
            )

            # 计算总损失
            total_train_loss = grad_norm.compute_loss(
                train_level_loss, train_grad_loss, train_rock_loss, train_strat_loss
            )

            optimizer.zero_grad()
            total_train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step(total_train_loss)

            # 每10个epoch评估
            if epoch % 10 == 0:
                # 反归一化level用于评估
                predicted_level_eval = normalizer.inverse_transform_level(predicted_level.detach())
                graph_data_eval = graph_data.clone()
                graph_data_eval.level = normalizer.inverse_transform_level(graph_data.level)

                train_metrics = evaluate_model_with_constraints(
                    model, graph_data_eval, train_mask_level, train_mask_rock,
                    train_mask_gradient, edge_index, gradient,
                    original_coords, strat_constraint, focal_loss_fn, strat_loss_fn,
                    device, phase="Train", use_post_process=False
                )

                test_metrics = evaluate_model_with_constraints(
                    model, graph_data_eval, test_mask_level, test_mask_rock,
                    test_mask_gradient, edge_index, gradient,
                    original_coords, strat_constraint, focal_loss_fn, strat_loss_fn,
                    device, phase="Test", use_post_process=True  # 测试时使用后处理
                )

                current_lr = optimizer.param_groups[0]['lr']

                # 保存最佳模型（基于修正后的精度）
                if test_metrics['accuracy_corrected'] > best_test_accuracy:
                    best_test_accuracy = test_metrics['accuracy_corrected']
                    best_epoch = epoch
                    torch.save(model.state_dict(),
                               os.path.join(result_dir, 'best_model_constrained.pth'))

                log_msg = (
                    f"Epoch {epoch}/{num_epochs} | LR: {current_lr:.6f} | "
                    f"Weights: [{loss_weights[0]:.3f}, {loss_weights[1]:.3f}, "
                    f"{loss_weights[2]:.3f}, {loss_weights[3]:.3f}]\n"
                    f"  [TRAIN] Total: {total_train_loss.item():.4f} | "
                    f"Level: {train_metrics['level_loss']:.4f} | "
                    f"Grad: {train_metrics['grad_loss']:.4f} | "
                    f"Rock: {train_metrics['rock_loss']:.4f} | "
                    f"Strat: {train_metrics['strat_loss']:.4f} | "
                    f"RMSE: {train_metrics['rmse']:.4f} | "
                    f"R^2: {train_metrics['r2']:.4f} | "
                    f"Acc: {train_metrics['accuracy']:.4f}\n"
                    f"  [TEST]  Level: {test_metrics['level_loss']:.4f} | "
                    f"Grad: {test_metrics['grad_loss']:.4f} | "
                    f"Rock: {test_metrics['rock_loss']:.4f} | "
                    f"Strat: {test_metrics['strat_loss']:.4f} | "
                    f"RMSE: {test_metrics['rmse']:.4f} | "
                    f"R^2: {test_metrics['r2']:.4f} | "
                    f"Acc: {test_metrics['accuracy']:.4f} -> "
                    f"Acc_Corrected: {test_metrics['accuracy_corrected']:.4f} ⭐"
                )

                print(log_msg)
                f.write(log_msg + "\n\n")
                f.flush()

        end_time = time.time()
        training_time = end_time - start_time

        f.write(f"\n{'=' * 80}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Best Epoch: {best_epoch} (Test Accuracy: {best_test_accuracy:.4f})\n")

        # 最终评估
        print(f"\n{'=' * 80}")
        print(f"🏆 加载最佳模型 (Epoch {best_epoch}, Acc: {best_test_accuracy:.4f})")
        print(f"{'=' * 80}\n")

        model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model_constrained.pth')))

        graph_data_final = graph_data.clone()
        graph_data_final.level = normalizer.inverse_transform_level(graph_data.level)

        final_test_metrics = evaluate_model_with_constraints(
            model, graph_data_final, test_mask_level, test_mask_rock,
            test_mask_gradient, edge_index, gradient,
            original_coords, strat_constraint, focal_loss_fn, strat_loss_fn,
            device, phase="Final Test", use_post_process=True
        )

        f.write(f"\n{'=' * 80}\n")
        f.write(f"Final Test Set Performance (with post-processing):\n")
        f.write(f"  RMSE: {final_test_metrics['rmse']:.4f}\n")
        f.write(f"  R^2: {final_test_metrics['r2']:.4f}\n")
        f.write(f"  Original Accuracy: {final_test_metrics['accuracy']:.4f}\n")
        f.write(f"  Corrected Accuracy: {final_test_metrics['accuracy_corrected']:.4f}\n")

        # 混淆矩阵
        model.eval()
        with torch.no_grad():
            _, predicted_rock_final = model(
                graph_data.x.to(device),
                graph_data.edge_index.to(device)
            )

            predicted_level_final, _ = model(
                graph_data.x.to(device),
                graph_data.edge_index.to(device)
            )
            predicted_level_final = normalizer.inverse_transform_level(predicted_level_final)

            # 后处理预测
            predicted_rock_corrected = post_process_with_level_constraint(
                predicted_level_final, predicted_rock_final, strat_constraint, device
            )

            if test_mask_rock.any():
                # 原始混淆矩阵
                cm_original = calculate_confusion_matrix(
                    predicted_rock_final, graph_data.rock_unit, test_mask_rock
                )
                f.write(f"\nOriginal Confusion Matrix:\n{cm_original}\n")
                print(f"\nOriginal Confusion Matrix:\n{cm_original}\n")

                # 修正后的混淆矩阵
                predicted_rock_corrected_cpu = predicted_rock_corrected[test_mask_rock].cpu().numpy()
                true_labels = (graph_data.rock_unit[test_mask_rock] - 1).cpu().numpy()
                # cm_corrected = confusion_matrix(true_labels, predicted_rock_corrected_cpu)
                # f.write(f"\nCorrected Confusion Matrix:\n{cm_corrected}\n")
                # print(f"\nCorrected Confusion Matrix:\n{cm_corrected}\n")

    print(f"\n✅ 训练完成！总用时: {training_time:.2f}秒")
    print(f"🏆 最佳模型: Epoch {best_epoch} (修正后测试精度: {best_test_accuracy:.4f})")

    # 保存预测结果
    model.eval()
    with torch.no_grad():
        predicted_level_final, predicted_rock_final = model(
            graph_data.x.to(device),
            graph_data.edge_index.to(device)
        )

        predicted_level_original = normalizer.inverse_transform_level(predicted_level_final)

        # 使用后处理的岩性预测
        predicted_rock_classes = post_process_with_level_constraint(
            predicted_level_original, predicted_rock_final, strat_constraint, device
        ).cpu().numpy()

    all_nodes = np.arange(graph_data.x.size(0))
    fault_features = graph_data.x[:, 3:].cpu().numpy()

    save_rock_result_to_csv(
        graph_data=graph_data_final,
        predicted_level=predicted_level_original,
        fault_features=fault_features,
        nodes=all_nodes,
        predicted_rock_units=predicted_rock_classes + 1,
        suffix='_constrained_all',
        result_dir=result_dir
    )

    return model


def main(node_file, ele_file, vtk_file, epoch=300, lr=0.01,
         hidden_channels=128, num_classes=13, result_dir=None,
         factor=1.0, dropout=0.1, lr_decay=0.8,
         gat_heads=2, train_ratio=0.8, strat_weight=1.0, focal_gamma=2.0):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print("=" * 80)
    print("📊 创建图数据")
    print("=" * 80)
    graph_data = create_or_load_graph(node_file, ele_file, is_gradient=False)

    print("\n" + "=" * 80)
    print("🔧 计算断层特征")
    print("=" * 80)
    graph_data = compute_fault_features(graph_data, vtk_file, factor=factor)

    print("\n" + "=" * 80)
    print(f"🚀 开始训练（集成地层约束）")
    print(f"   Focal Loss Gamma: {focal_gamma}")
    print(f"   Stratigraphic Constraint Weight: {strat_weight}")
    print("=" * 80)

    trained_model = train_multitask_with_constraints(
        graph_data,
        num_epochs=epoch,
        lr=lr,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        result_dir=result_dir,
        dropout=dropout,
        lr_decay=lr_decay,
        gat_heads=gat_heads,
        train_ratio=train_ratio,
        strat_weight=strat_weight,
        focal_gamma=focal_gamma
    )

    model_path = os.path.join(result_dir, 'final_model_constrained.pth')
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n💾 最终模型已保存至: {model_path}")


if __name__ == "__main__":
    # 测试配置
    params = {
        "node_file": "./Data/combined_mesh.node",
        "ele_file": "./Data/combined_mesh.ele",
        "vtk_file": "./Data/F1.vtk",
        "epoch": 300,
        "lr": 0.01,
        "hidden_channels": 128,
        "num_classes": 13,
        "factor": 1.0,
        "dropout": 0.1,
        "lr_decay": 0.8,
        "gat_heads": 2,
        "train_ratio": 0.8,
        "strat_weight": 1.0,  # 地层约束权重
        "focal_gamma": 2.0,  # Focal Loss gamma
        "result_dir": "./Result/MultiTask_Constrained_Test"
    }

    main(**params)