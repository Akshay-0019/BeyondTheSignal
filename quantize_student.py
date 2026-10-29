import os
import argparse
import torch

from src.dataset import get_dataloaders
from src.models.student import get_student_model
from src.models.quantization import quantize_student_model, save_quantized_model, get_model_size_mb
from src.utils import measure_cpu_latency_ms, compute_metrics


def run_quantization(
    student_fp32_path='checkpoints/student_distilled_noise.pth',
    output_int8_path='checkpoints/student_int8.pth',
    num_calibration_batches=20
):
    print("==================================================")
    print(" Phase 3: Post-Training Static Quantization (PTQ)")
    print("==================================================")

    if not os.path.exists(student_fp32_path):
        raise FileNotFoundError(f"Trained student checkpoint not found at {student_fp32_path}!")

    # 1. Load trained FP32 student model
    fp32_size = get_model_size_mb(student_fp32_path)
    student_fp32 = get_student_model(num_classes=10)
    student_fp32.load_state_dict(torch.load(student_fp32_path, map_location='cpu', weights_only=True))
    student_fp32.eval()
    print(f"[FP32 Baseline] Checkpoint file size: {fp32_size:.3f} MB")

    # Measure FP32 CPU latency
    lat_fp32 = measure_cpu_latency_ms(student_fp32, num_warmup=20, num_iter=100)
    print(f"[FP32 Baseline] CPU Latency: {lat_fp32['mean_latency_ms']:.2f} ms ± {lat_fp32['std_latency_ms']:.2f} ms")

    # 2. Calibration DataLoader (Representative Satellite Patches)
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_dir='./data/eurosat_2000',
        batch_size=32,
        use_noise_augmentation=False
    )

    # 3. Apply Static Post-Training Quantization
    quantized_student = quantize_student_model(
        student_fp32,
        calibration_loader=train_loader,
        num_calibration_batches=num_calibration_batches
    )

    # 4. Save INT8 Model
    int8_size = save_quantized_model(quantized_student, output_int8_path)

    # 5. Measure INT8 CPU Latency
    lat_int8 = measure_cpu_latency_ms(quantized_student, num_warmup=20, num_iter=100)
    print(f"[INT8 Quantized] CPU Latency: {lat_int8['mean_latency_ms']:.2f} ms ± {lat_int8['std_latency_ms']:.2f} ms")

    # 6. Sanity check test accuracy
    print("[Quantization] Evaluating INT8 test accuracy vs FP32 test accuracy...")
    test_preds_fp32, test_preds_int8, test_targets = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images_cpu = images.to('cpu')
            out_fp32 = student_fp32(images_cpu)
            out_int8 = quantized_student(images_cpu)

            test_preds_fp32.extend(torch.argmax(out_fp32, dim=1).numpy())
            test_preds_int8.extend(torch.argmax(out_int8, dim=1).numpy())
            test_targets.extend(labels.numpy())

    m_fp32 = compute_metrics(test_targets, test_preds_fp32)
    m_int8 = compute_metrics(test_targets, test_preds_int8)

    print("\n---------------- PTQ Summary Report ----------------")
    print(f"FP32 Model Size     : {fp32_size:.3f} MB")
    print(f"INT8 Model Size     : {int8_size:.3f} MB")
    print(f"Footprint Reduction : {fp32_size / int8_size:.2f}x reduction ({((1 - int8_size/fp32_size)*100):.1f}% smaller)")
    print(f"FP32 Latency (CPU)  : {lat_fp32['mean_latency_ms']:.2f} ms")
    print(f"INT8 Latency (CPU)  : {lat_int8['mean_latency_ms']:.2f} ms")
    print(f"CPU Speedup Factor  : {lat_fp32['mean_latency_ms'] / lat_int8['mean_latency_ms']:.2f}x speedup")
    print(f"FP32 Test F1-Score  : {m_fp32['macro_f1']:.4f}")
    print(f"INT8 Test F1-Score  : {m_int8['macro_f1']:.4f} (Accuracy Retention: {(m_int8['macro_f1'] / m_fp32['macro_f1']) * 100:.1f}%)")
    print("----------------------------------------------------\n")

    return output_int8_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Static INT8 Quantization of MicroSatNet")
    parser.add_argument('--fp32-checkpoint', type=str, default='checkpoints/student_distilled_noise.pth')
    parser.add_argument('--output-int8', type=str, default='checkpoints/student_int8.pth')
    parser.add_argument('--calib-batches', type=int, default=20, dest='calib_batches')
    args = parser.parse_args()

    run_quantization(
        student_fp32_path=args.fp32_checkpoint,
        output_int8_path=args.output_int8,
        num_calibration_batches=args.calib_batches
    )
