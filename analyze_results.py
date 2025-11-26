"""
结果分析和可视化脚本
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import re


def parse_training_log(log_file):
    """
    解析训练日志，提取关键指标

    返回:
    - metrics_df: DataFrame包含每个epoch的指标
    """
    if not os.path.exists(log_file):
        print(f"⚠️  日志文件不存在: {log_file}")
        return None

    data = {
        'epoch': [],
        'train_level_loss': [],
        'train_grad_loss': [],
        'train_rock_loss': [],
        'train_strat_loss': [],
        'train_rmse': [],
        'train_r2': [],
        'train_acc': [],
        'test_level_loss': [],
        'test_grad_loss': [],
        'test_rock_loss': [],
        'test_strat_loss': [],
        'test_rmse': [],
        'test_r2': [],
        'test_acc': [],
        'test_acc_corrected': []
    }

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = None

    for i, line in enumerate(lines):
        # 提取epoch
        if 'Epoch' in line and '/' in line:
            try:
                epoch = int(line.split('Epoch')[1].split('/')[0].strip())
                current_epoch = epoch
            except:
                continue

        # 提取训练指标
        if '[TRAIN]' in line and current_epoch:
            try:
                data['epoch'].append(current_epoch)

                # 提取各项损失
                level = float(re.search(r'Level: ([\d.]+)', line).group(1))
                grad = float(re.search(r'Grad: ([\d.]+)', line).group(1))
                rock = float(re.search(r'Rock: ([\d.]+)', line).group(1))

                data['train_level_loss'].append(level)
                data['train_grad_loss'].append(grad)
                data['train_rock_loss'].append(rock)

                # 地层约束损失（如果有）
                if 'Strat:' in line:
                    strat = float(re.search(r'Strat: ([\d.]+)', line).group(1))
                    data['train_strat_loss'].append(strat)
                else:
                    data['train_strat_loss'].append(0.0)

                # RMSE, R^2, Acc
                rmse = float(re.search(r'RMSE: ([\d.]+)', line).group(1))
                r2 = float(re.search(r'R\^2: ([\d.]+)', line).group(1))
                acc = float(re.search(r'Acc: ([\d.]+)', line).group(1))

                data['train_rmse'].append(rmse)
                data['train_r2'].append(r2)
                data['train_acc'].append(acc)
            except:
                # 如果解析失败，移除当前epoch
                if data['epoch'] and data['epoch'][-1] == current_epoch:
                    for key in data:
                        if data[key] and len(data[key]) > 0:
                            data[key].pop()

        # 提取测试指标
        if '[TEST]' in line and current_epoch:
            try:
                level = float(re.search(r'Level: ([\d.]+)', line).group(1))
                grad = float(re.search(r'Grad: ([\d.]+)', line).group(1))
                rock = float(re.search(r'Rock: ([\d.]+)', line).group(1))

                data['test_level_loss'].append(level)
                data['test_grad_loss'].append(grad)
                data['test_rock_loss'].append(rock)

                if 'Strat:' in line:
                    strat = float(re.search(r'Strat: ([\d.]+)', line).group(1))
                    data['test_strat_loss'].append(strat)
                else:
                    data['test_strat_loss'].append(0.0)

                rmse = float(re.search(r'RMSE: ([\d.]+)', line).group(1))
                r2 = float(re.search(r'R\^2: ([\d.]+)', line).group(1))
                acc = float(re.search(r'Acc: ([\d.]+)', line).group(1))

                data['test_rmse'].append(rmse)
                data['test_r2'].append(r2)
                data['test_acc'].append(acc)

                # 修正后的精度
                if 'Acc_Corrected:' in line:
                    acc_corr = float(re.search(r'Acc_Corrected: ([\d.]+)', line).group(1))
                    data['test_acc_corrected'].append(acc_corr)
                else:
                    data['test_acc_corrected'].append(acc)
            except:
                pass

    # 创建DataFrame
    df = pd.DataFrame(data)
    return df


def plot_training_curves(metrics_df, save_dir='.'):
    """
    绘制训练曲线
    """
    if metrics_df is None or len(metrics_df) == 0:
        print("⚠️  没有数据可绘制")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    # 1. 损失曲线
    ax = axes[0, 0]
    ax.plot(metrics_df['epoch'], metrics_df['train_level_loss'],
            label='Train Level', marker='o', markersize=3)
    ax.plot(metrics_df['epoch'], metrics_df['test_level_loss'],
            label='Test Level', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Level Loss')
    ax.set_title('Level Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 梯度损失
    ax = axes[0, 1]
    ax.plot(metrics_df['epoch'], metrics_df['train_grad_loss'],
            label='Train Gradient', marker='o', markersize=3)
    ax.plot(metrics_df['epoch'], metrics_df['test_grad_loss'],
            label='Test Gradient', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Loss')
    ax.set_title('Gradient Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 岩性损失
    ax = axes[0, 2]
    ax.plot(metrics_df['epoch'], metrics_df['train_rock_loss'],
            label='Train Rock', marker='o', markersize=3)
    ax.plot(metrics_df['epoch'], metrics_df['test_rock_loss'],
            label='Test Rock', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rock Loss')
    ax.set_title('Rock (Focal) Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 地层约束损失
    ax = axes[1, 0]
    if 'train_strat_loss' in metrics_df.columns:
        ax.plot(metrics_df['epoch'], metrics_df['train_strat_loss'],
                label='Train Strat', marker='o', markersize=3)
        ax.plot(metrics_df['epoch'], metrics_df['test_strat_loss'],
                label='Test Strat', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Stratigraphic Loss')
        ax.set_title('Stratigraphic Constraint Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 5. RMSE
    ax = axes[1, 1]
    ax.plot(metrics_df['epoch'], metrics_df['train_rmse'],
            label='Train RMSE', marker='o', markersize=3)
    ax.plot(metrics_df['epoch'], metrics_df['test_rmse'],
            label='Test RMSE', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 岩性分类精度
    ax = axes[1, 2]
    ax.plot(metrics_df['epoch'], metrics_df['train_acc'],
            label='Train Acc', marker='o', markersize=3)
    ax.plot(metrics_df['epoch'], metrics_df['test_acc'],
            label='Test Acc (Original)', marker='s', markersize=3)

    if 'test_acc_corrected' in metrics_df.columns:
        ax.plot(metrics_df['epoch'], metrics_df['test_acc_corrected'],
                label='Test Acc (Corrected)', marker='^', markersize=3, linewidth=2)

    # 添加91%目标线
    ax.axhline(y=0.91, color='r', linestyle='--', linewidth=2,
               label='Target (91%)', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Rock Unit Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {save_path}")

    plt.show()


def plot_confusion_matrix_from_log(log_file, save_dir='.'):
    """
    从日志文件中提取并绘制混淆矩阵
    """
    if not os.path.exists(log_file):
        print(f"⚠️  日志文件不存在: {log_file}")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取修正后的混淆矩阵
    if 'Corrected Confusion Matrix:' in content:
        cm_text = content.split('Corrected Confusion Matrix:')[1].split('\n\n')[0]
        matrix_type = 'Corrected'
    elif 'Original Confusion Matrix:' in content:
        cm_text = content.split('Original Confusion Matrix:')[1].split('\n\n')[0]
        matrix_type = 'Original'
    else:
        print("⚠️  日志中未找到混淆矩阵")
        return

    # 解析矩阵
    try:
        lines = [line.strip() for line in cm_text.strip().split('\n') if line.strip()]
        cm = []
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                row = [int(x) for x in line.strip('[]').split()]
                cm.append(row)

        cm = np.array(cm)

        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))

        # 计算每行的准确率
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / row_sums

        # 绘制热力图
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    cbar_kws={'label': 'Accuracy'})

        plt.xlabel('Predicted Rock Unit', fontsize=12)
        plt.ylabel('True Rock Unit', fontsize=12)
        plt.title(f'{matrix_type} Confusion Matrix (Normalized by Row)',
                  fontsize=14, fontweight='bold')

        # 添加类别标签
        rock_units = list(range(1, cm.shape[0] + 1))
        plt.xticks(np.arange(len(rock_units)) + 0.5, rock_units)
        plt.yticks(np.arange(len(rock_units)) + 0.5, rock_units)

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f'confusion_matrix_{matrix_type.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存: {save_path}")

        plt.show()

        # 计算并打印各类别精度
        print(f"\n各岩性类别精度 ({matrix_type}):")
        for i, rock_unit in enumerate(rock_units):
            if row_sums[i] > 0:
                acc = cm[i, i] / row_sums[i]
                print(f"  Rock Unit {rock_unit:2d}: {acc:.2%} ({cm[i, i]}/{int(row_sums[i])})")

        overall_acc = np.trace(cm) / cm.sum()
        print(f"\n总体精度: {overall_acc:.4f} ({overall_acc:.2%})")

    except Exception as e:
        print(f"❌ 解析混淆矩阵失败: {e}")


def compare_experiments(result_dirs, labels=None):
    """
    比较多个实验结果

    参数:
    - result_dirs: 结果目录列表
    - labels: 实验标签列表（可选）
    """
    if labels is None:
        labels = [f"Exp{i + 1}" for i in range(len(result_dirs))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 测试精度对比
    ax = axes[0]

    for i, (result_dir, label) in enumerate(zip(result_dirs, labels)):
        log_file = os.path.join(result_dir, 'training_log_with_constraints.txt')
        if not os.path.exists(log_file):
            log_file = os.path.join(result_dir, 'multitask_training_log_v1.txt')

        if os.path.exists(log_file):
            df = parse_training_log(log_file)
            if df is not None and len(df) > 0:
                if 'test_acc_corrected' in df.columns:
                    ax.plot(df['epoch'], df['test_acc_corrected'],
                            label=label, marker='o', markersize=3)
                else:
                    ax.plot(df['epoch'], df['test_acc'],
                            label=label, marker='o', markersize=3)

    ax.axhline(y=0.91, color='r', linestyle='--', linewidth=2,
               label='Target (91%)', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])

    # 2. 最终精度条形图
    ax = axes[1]

    final_accs = []
    for result_dir in result_dirs:
        log_file = os.path.join(result_dir, 'training_log_with_constraints.txt')
        if not os.path.exists(log_file):
            log_file = os.path.join(result_dir, 'multitask_training_log_v1.txt')

        if os.path.exists(log_file):
            df = parse_training_log(log_file)
            if df is not None and len(df) > 0:
                if 'test_acc_corrected' in df.columns:
                    final_accs.append(df['test_acc_corrected'].max())
                else:
                    final_accs.append(df['test_acc'].max())
            else:
                final_accs.append(0)
        else:
            final_accs.append(0)

    colors = ['green' if acc >= 0.91 else 'orange' for acc in final_accs]
    bars = ax.bar(labels, final_accs, color=colors, alpha=0.7)
    ax.axhline(y=0.91, color='r', linestyle='--', linewidth=2, label='Target')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Final Accuracy Comparison')
    ax.set_ylim([0.7, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 实验对比图已保存: experiment_comparison.png")
    plt.show()


def analyze_result(result_dir):
    """
    全面分析单个实验结果
    """
    print(f"\n{'=' * 80}")
    print(f"📊 分析实验结果: {result_dir}")
    print(f"{'=' * 80}\n")

    # 1. 解析训练日志
    log_file = os.path.join(result_dir, 'training_log_with_constraints.txt')
    if not os.path.exists(log_file):
        log_file = os.path.join(result_dir, 'multitask_training_log_v1.txt')

    if os.path.exists(log_file):
        print("✅ 找到训练日志")
        df = parse_training_log(log_file)

        if df is not None and len(df) > 0:
            print(f"   训练了 {len(df)} 个记录点")

            # 打印最终结果
            if 'test_acc_corrected' in df.columns:
                best_acc = df['test_acc_corrected'].max()
                best_epoch = df.loc[df['test_acc_corrected'].idxmax(), 'epoch']
            else:
                best_acc = df['test_acc'].max()
                best_epoch = df.loc[df['test_acc'].idxmax(), 'epoch']

            print(f"   最佳精度: {best_acc:.4f} (Epoch {int(best_epoch)})")

            if best_acc >= 0.91:
                print("   🎉 达到目标精度 (>91%)!")
            else:
                diff = 0.91 - best_acc
                print(f"   ⚠️  距离目标还差 {diff:.4f} ({diff * 100:.2f}%)")

            # 绘制训练曲线
            plot_training_curves(df, save_dir=result_dir)

        # 绘制混淆矩阵
        plot_confusion_matrix_from_log(log_file, save_dir=result_dir)

    else:
        print("❌ 未找到训练日志")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='结果分析脚本')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='结果目录路径')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='比较多个实验目录')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='实验标签（与--compare配合使用）')

    args = parser.parse_args()

    if args.compare:
        # 比较模式
        compare_experiments(args.compare, args.labels)
    elif args.result_dir:
        # 单个实验分析
        analyze_result(args.result_dir)
    else:
        # 默认：分析最新的结果
        print("请指定 --result_dir 或 --compare 参数")
        print("\n示例:")
        print("  python analyze_results.py --result_dir ./Result/MultiTask_Constrained_Test")
        print("  python analyze_results.py --compare ./Result/Exp1 ./Result/Exp2 --labels Original Improved")