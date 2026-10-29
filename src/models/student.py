import torch
import torch.nn as nn
import torch.ao.quantization as quantization


class SqueezeExcite(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention (MobileNetV3 style)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return self.mul.mul(x, scale)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block using Depthwise-Separable Convolutions
    with optional Squeeze-and-Excitation attention.
    Compatible with PyTorch Static Quantization (PTQ).
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=3, use_se=True):
        super().__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))
        self.has_expansion = (hidden_dim != in_channels)
        self.use_se = use_se

        layers = []
        if self.has_expansion:
            # 1x1 Expansion Conv
            self.conv_exp = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn_exp = nn.BatchNorm2d(hidden_dim)
            self.relu_exp = nn.ReLU(inplace=False)

        # 3x3 Depthwise Conv
        self.conv_dw = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride,
            padding=1, groups=hidden_dim, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(hidden_dim)
        self.relu_dw = nn.ReLU(inplace=False)

        # Squeeze-and-Excitation
        if self.use_se:
            self.se = SqueezeExcite(hidden_dim, reduction=4)

        # 1x1 Pointwise Linear Projection
        self.conv_pw = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)

        if self.use_residual:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = x
        if self.has_expansion:
            out = self.relu_exp(self.bn_exp(self.conv_exp(out)))
        out = self.relu_dw(self.bn_dw(self.conv_dw(out)))
        if self.use_se:
            out = self.se(out)
        out = self.bn_pw(self.conv_pw(out))

        if self.use_residual:
            out = self.skip_add.add(residual, out)
        return out

    def fuse_block(self):
        """Fuses Conv + BN + ReLU modules for static quantization."""
        if self.has_expansion:
            quantization.fuse_modules(self, [['conv_exp', 'bn_exp', 'relu_exp']], inplace=True)
        quantization.fuse_modules(self, [['conv_dw', 'bn_dw', 'relu_dw']], inplace=True)
        quantization.fuse_modules(self, [['conv_pw', 'bn_pw']], inplace=True)


class MicroSatNet(nn.Module):
    """
    Ultra-compact Depthwise-Separable CNN with Squeeze-and-Excitation attention,
    designed for CubeSat onboard inference. Uses MobileNetV3-style inverted residuals.
    Parameter footprint: ~360K parameters (< 1.5 MB in FP32, ~0.4 MB in INT8).
    Fully compatible with PyTorch static INT8 quantization.
    """
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.quant = quantization.QuantStub()

        # Initial stem: 3 -> 32
        self.stem_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU(inplace=False)

        # Inverted Residual depthwise-separable stages
        self.block1 = InvertedResidualBlock(32, 32, stride=1, expand_ratio=1)
        self.block2 = InvertedResidualBlock(32, 48, stride=2, expand_ratio=3)  # 64 -> 32
        self.block3 = InvertedResidualBlock(48, 48, stride=1, expand_ratio=3)
        self.block4 = InvertedResidualBlock(48, 64, stride=2, expand_ratio=3)  # 32 -> 16
        self.block5 = InvertedResidualBlock(64, 64, stride=1, expand_ratio=3)
        self.block6 = InvertedResidualBlock(64, 96, stride=2, expand_ratio=3)  # 16 -> 8
        self.block7 = InvertedResidualBlock(96, 128, stride=1, expand_ratio=3)

        # Head
        self.head_conv = nn.Conv2d(128, 160, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(160)
        self.head_relu = nn.ReLU(inplace=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(160, num_classes)

        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.head_relu(self.head_bn(self.head_conv(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuses all Conv+BN+ReLU layers across stem, blocks, and head."""
        quantization.fuse_modules(self, [['stem_conv', 'stem_bn', 'stem_relu']], inplace=True)
        self.block1.fuse_block()
        self.block2.fuse_block()
        self.block3.fuse_block()
        self.block4.fuse_block()
        self.block5.fuse_block()
        self.block6.fuse_block()
        self.block7.fuse_block()
        quantization.fuse_modules(self, [['head_conv', 'head_bn', 'head_relu']], inplace=True)


def get_student_model(num_classes=10):
    return MicroSatNet(num_classes=num_classes)


if __name__ == '__main__':
    model = get_student_model(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    print(f"MicroSatNet Total Trainable Parameters: {total_params:,}")
    print(f"MicroSatNet Parameter Footprint (FP32): {fp32_size_mb:.3f} MB (Budget: < 5.0 MB)")

    dummy = torch.randn(2, 3, 64, 64)
    out = model(dummy)
    print(f"Output shape for [2, 3, 64, 64]: {out.shape}")
