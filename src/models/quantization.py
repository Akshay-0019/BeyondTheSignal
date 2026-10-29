import os
import copy
import torch
import torch.ao.quantization as quantization
import torchvision.transforms as transforms
from src.models.student import get_student_model


def get_quantization_engine():
    """Detects best available engine: qnnpack (ARM/Mac) or fbgemm (x86)."""
    supported = torch.backends.quantized.supported_engines
    if 'qnnpack' in supported:
        return 'qnnpack'
    elif 'fbgemm' in supported:
        return 'fbgemm'
    return 'qnnpack'


def quantize_student_model(model_fp32, calibration_loader, num_calibration_batches=15):
    """
    Applies PyTorch Static Post-Training Quantization (PTQ) to convert FP32 weights & activations to INT8.
    """
    engine = get_quantization_engine()
    torch.backends.quantized.engine = engine
    print(f"[Quantization] Target quantization engine: {engine}")

    norm_transform = transforms.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.2024, 0.1368, 0.1157])

    # Prepare CPU copy in eval mode
    model_cpu = copy.deepcopy(model_fp32).to('cpu').eval()

    # Fuse Conv + BN + ReLU modules
    print("[Quantization] Fusing Conv+BatchNorm+ReLU modules...")
    model_cpu.fuse_model()

    # Assign static quantization configuration
    model_cpu.qconfig = quantization.get_default_qconfig(engine)
    prepared_model = quantization.prepare(model_cpu, inplace=False)

    # Calibration step with representative satellite patches
    print(f"[Quantization] Running calibration over {num_calibration_batches} batches...")
    prepared_model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            if images.min() >= 0.0 and images.max() <= 1.0:
                images_cpu = norm_transform(images).to('cpu')
            else:
                images_cpu = images.to('cpu')
            prepared_model(images_cpu)

    # Convert calibrated model to INT8
    print("[Quantization] Converting model to INT8 representation...")
    quantized_model = quantization.convert(prepared_model, inplace=False)
    print("[Quantization] PTQ static quantization completed successfully!")
    return quantized_model


def save_quantized_model(quantized_model, save_path):
    """
    Saves the quantized model state dict or full TorchScript package.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[Quantization] Saved INT8 quantized model to {save_path} ({file_size_mb:.3f} MB)")
    return file_size_mb


def load_quantized_model(load_path, num_classes=10):
    """
    Instantiates, fuses, prepares, converts, and loads state dict for INT8 inference.
    """
    engine = get_quantization_engine()
    torch.backends.quantized.engine = engine

    base_model = get_student_model(num_classes=num_classes).to('cpu').eval()
    base_model.fuse_model()
    base_model.qconfig = quantization.get_default_qconfig(engine)
    prepared = quantization.prepare(base_model, inplace=False)
    quantized = quantization.convert(prepared, inplace=False)
    quantized.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
    quantized.eval()
    return quantized


def get_model_size_mb(filepath):
    """Uses os.path.getsize() to calculate exact footprint in Megabytes."""
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / (1024 * 1024)
