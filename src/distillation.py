import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining Cross-Entropy with ground-truth
    and Kullback-Leibler (KL) divergence with soft teacher probabilities.

    L_KD = alpha * L_CE(z_s, y) + (1 - alpha) * T^2 * KL(softmax(z_s / T), softmax(z_t / T))
    """
    def __init__(self, temperature=4.0, alpha=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss (Cross Entropy with ground-truth)
        loss_ce = self.ce_loss(student_logits, labels)

        # Soft label loss (KL divergence scaled by T^2)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        loss_soft = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        # Total combined distillation loss
        total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_soft
        return total_loss, loss_ce, loss_soft


if __name__ == '__main__':
    criterion = DistillationLoss(temperature=4.0, alpha=0.3)
    s_logits = torch.randn(8, 10, requires_grad=True)
    t_logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    loss, ce, soft = criterion(s_logits, t_logits, targets)
    loss.backward()
    print(f"Distillation test: total={loss.item():.4f}, ce={ce.item():.4f}, soft={soft.item():.4f}")
