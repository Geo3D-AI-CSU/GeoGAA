from sklearn.metrics import mean_squared_error,r2_score, confusion_matrix
import numpy as np



# RMSE计算
def calculate_rmse(predicted, target, mask):
    mask =mask.to('cpu')  # 将 mask 转到 CPU 上
    predicted = predicted[mask].detach().cpu().numpy()  # 使用 .detach() 来分离计算图
    target = target[mask].detach().cpu().numpy()  # 同样分离 target
    return np.sqrt(mean_squared_error(target, predicted))


# 准确率计算
def calculate_accuracy(predicted, target, mask):
    # 确保所有张量在相同设备上
    mask = mask.to('cpu')
    predicted = predicted[mask].detach().cpu()  # 去掉梯度信息并转到 CPU 上
    target = target[mask].detach().cpu()-1

    # 获取预测的类别（每行的最大值的索引）
    predicted_classes = predicted.argmax(dim=1)  # 选择每个节点的最大概率对应的类别
    correct = (predicted_classes == target).sum().item()  # 计算预测正确的数量
    total = target.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# R 方计算
def calculate_r2(predicted, target, mask):
    # 从 mask 中选择所需的值，并分离计算图
    predicted = predicted[mask].detach().cpu().numpy()
    target = target[mask].detach().numpy()
    return r2_score(target, predicted)

# 混淆矩阵计算
def calculate_confusion_matrix(predicted, target, mask):
    # 通过 mask 筛选出对应的节点
    mask = mask.to('cpu')
    predicted = predicted[mask].detach().cpu().numpy()
    target = target[mask].detach().cpu().numpy()-1
    predicted_classes = predicted.argmax(axis=1)
    # 计算混淆矩阵
    return confusion_matrix(target,  predicted_classes)