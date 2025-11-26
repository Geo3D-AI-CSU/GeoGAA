import torch

class GradNorm_2loss:
    def __init__(self, alpha=1.0, gamma=1.0, device='cuda'):
        """
        初始化 GradNorm 计算器，考虑 level_loss 和 grad_loss
        alpha, gamma 是初始权重。
        """
        self.alpha = alpha  # Level loss 初始权重
        self.gamma = gamma  # Gradient loss 初始权重
        self.device = device
        # 权重存储在一个张量中
        self.loss_weights = torch.tensor([alpha, gamma], dtype=torch.float32, device=device)

    def compute_loss(self, level_loss, gradient_loss):
        """
        计算加权总损失，基于更新的损失权重。
        """
        return self.loss_weights[0] * level_loss + self.loss_weights[1] * gradient_loss

    def update_weights(self, level_loss, gradient_loss, model):
        """
        根据 GradNorm 算法动态更新损失函数的权重。
        """
        # 计算每个损失的梯度范数
        level_grad_norm = self.compute_grad_norm(level_loss, model)
        grad_grad_norm = self.compute_grad_norm(gradient_loss, model)

        grad_norms = torch.tensor([level_grad_norm, grad_grad_norm], device=self.device)

        # 归一化梯度范数
        normed_grad_norms = grad_norms / grad_norms.mean()
        grad_ratio = normed_grad_norms / (grad_norms + 1e-8)  # 防止除零

        # 更新权重
        self.loss_weights[:len(grad_ratio)] = self.loss_weights[:len(grad_ratio)] * grad_ratio
        self.loss_weights = self.loss_weights / self.loss_weights.sum()  # 归一化权重，使其总和为1

        return self.loss_weights

    def compute_grad_norm(self, loss, model):
        """
        计算每个损失的梯度范数。
        """
        loss.backward(retain_graph=True)  # 保持计算图
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5

class GradNorm_3loss:
    def __init__(self, alpha=1.0, gamma=1.0, delta=1.0, device='cuda'):
        """
        初始化 GradNorm 计算器，考虑多个损失函数的权重
        alpha, gamma, delta 是初始权重，分别对应三个损失函数。
        """
        self.alpha = alpha  # 第一个损失函数的初始权重
        self.gamma = gamma  # 第二个损失函数的初始权重
        self.delta = delta  # 第三个损失函数的初始权重
        self.device = device
        # 权重存储在一个张量中，初始化为 [alpha, gamma, delta]
        self.loss_weights = torch.tensor([alpha, gamma, delta], dtype=torch.float32, device=device)

    def compute_loss(self, level_loss, gradient_loss, scalar_loss):
        """
        计算加权总损失，基于更新的损失权重。
        """
        # 这里加入三个损失函数
        return self.loss_weights[0] * level_loss + self.loss_weights[1] * gradient_loss + self.loss_weights[2] * scalar_loss

    def update_weights(self, level_loss, gradient_loss, scalar_loss, model):
        """
        根据 GradNorm 算法动态更新损失函数的权重。
        """
        # 计算每个损失的梯度范数
        level_grad_norm = self.compute_grad_norm(level_loss, model)
        grad_grad_norm = self.compute_grad_norm(gradient_loss, model)
        scalar_grad_norm = self.compute_grad_norm(scalar_loss, model)

        grad_norms = torch.tensor([level_grad_norm, grad_grad_norm, scalar_grad_norm], device=self.device)

        # # 打印梯度范数
        # print(
        #     f"Level Grad Norm: {level_grad_norm:.4f}, Gradient Grad Norm: {grad_grad_norm:.4f}, Scalar Grad Norm: {scalar_grad_norm:.4f}")

        # 归一化梯度范数
        normed_grad_norms = grad_norms / grad_norms.mean()
        grad_ratio = normed_grad_norms / (grad_norms + 1e-8)  # 防止除零

        # 更新权重
        self.loss_weights[:len(grad_ratio)] = self.loss_weights[:len(grad_ratio)] * grad_ratio
        self.loss_weights = self.loss_weights / self.loss_weights.sum()  # 归一化权重，使其总和为1

        return self.loss_weights

    def compute_grad_norm(self, loss, model):
        """
        计算每个损失的梯度范数。
        """
        loss.backward(retain_graph=True)  # 保持计算图
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5
