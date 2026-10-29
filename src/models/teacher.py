import torch
import torch.nn as nn
import torchvision.models as models


class TeacherNet(nn.Module):
    """
    High-capacity Teacher Network based on ResNet-50.
    Parameter count: ~23.5 Million parameters (~95 MB footprint).
    Acts as the high-accuracy feature extractor for Knowledge Distillation.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_teacher_model(num_classes=10, pretrained=True):
    return TeacherNet(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    teacher = get_teacher_model(num_classes=10, pretrained=False)
    total_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    print(f"TeacherNet (ResNet-50) Total Parameters: {total_params:,}")
    print(f"TeacherNet Parameter Footprint (FP32): {fp32_size_mb:.2f} MB")

    dummy = torch.randn(2, 3, 64, 64)
    out = teacher(dummy)
    print(f"Output shape for [2, 3, 64, 64]: {out.shape}")
