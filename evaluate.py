import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    SatelliteDataset,
    EUROSAT_CLASSES,
    GaussianNoiseTransform,
    SaltAndPepperNoiseTransform,
    BandDropoutTransform
)
from src.models.teacher import get_teacher_model
from src.models.student import get_student_model
from src.models.quantization import load_quantized_model, get_model_size_mb
from src.utils import measure_cpu_latency_ms, compute_metrics
from src.visualize import plot_f1_vs_noise, plot_latency_accuracy_tradeoff, plot_confusion_matrix


class FixedNoiseEvaluator:
    """Applies exact fixed synthetic noise levels to test batches for benchmarking."""
    def __init__(self, noise_type='clean', param=0.0):
        self.noise_type = noise_type
        self.param = param

        if noise_type == 'Gaussian':
            self.transform = GaussianNoiseTransform(std_range=(param, param), p=1.0)
        elif noise_type == 'Salt & Pepper':
            self.transform = SaltAndPepperNoiseTransform(amount_range=(param, param), p=1.0)
        elif noise_type == 'Band Dropout':
            self.transform = BandDropoutTransform(max_bands=int(param), p=1.0)
        else:
            self.transform = None

    def apply(self, images_unnorm, norm_transform):
        if self.transform is not None:
            noisy = torch.stack([self.transform(img) for img in images_unnorm])
        else:
            noisy = images_unnorm
        # Apply standard EuroSAT normalization
        return torch.stack([norm_transform(img) for img in noisy])


def run_evaluation(
    data_dir='./data/eurosat_2000',
    deliverables_dir='deliverables',
    checkpoints_dir='checkpoints'
):
    print("==================================================")
    print(" Phase 4: Headless Benchmarking & Metric Generation")
    print("==================================================")
    os.makedirs(deliverables_dir, exist_ok=True)

    # Prepare unnormalized test dataset so noise is injected into raw sensor signals
    test_ds = SatelliteDataset(os.path.join(data_dir, 'test'), dynamic_noise=None, normalize=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 1. Discover and load candidate models
    models_to_test = {}

    # Teacher
    teacher_pth = os.path.join(checkpoints_dir, 'teacher_best.pth')
    if os.path.exists(teacher_pth):
        t_model = get_teacher_model(num_classes=10, pretrained=False).to('cpu').eval()
        t_model.load_state_dict(torch.load(teacher_pth, map_location='cpu', weights_only=True))
        models_to_test['Teacher (ResNet-50 FP32)'] = {
            'model': t_model,
            'type': 'FP32_Teacher',
            'path': teacher_pth
        }

    # Student Baseline (Scratch)
    scratch_pth = os.path.join(checkpoints_dir, 'student_scratch.pth')
    if os.path.exists(scratch_pth):
        s_scratch = get_student_model(num_classes=10).to('cpu').eval()
        s_scratch.load_state_dict(torch.load(scratch_pth, map_location='cpu', weights_only=True))
        models_to_test['Student Baseline (Scratch FP32)'] = {
            'model': s_scratch,
            'type': 'FP32_Student_Scratch',
            'path': scratch_pth
        }

    # Student Distilled Clean (optional ablation)
    clean_pth = os.path.join(checkpoints_dir, 'student_distilled_clean.pth')
    if os.path.exists(clean_pth):
        s_clean = get_student_model(num_classes=10).to('cpu').eval()
        s_clean.load_state_dict(torch.load(clean_pth, map_location='cpu', weights_only=True))
        models_to_test['Student (Distilled Clean FP32)'] = {
            'model': s_clean,
            'type': 'FP32_Student_Clean',
            'path': clean_pth
        }

    # Student Distilled + Noise Trained
    distill_pth = os.path.join(checkpoints_dir, 'student_distilled_noise.pth')
    if os.path.exists(distill_pth):
        s_distill = get_student_model(num_classes=10).to('cpu').eval()
        s_distill.load_state_dict(torch.load(distill_pth, map_location='cpu', weights_only=True))
        models_to_test['Student (Distilled+Noise FP32)'] = {
            'model': s_distill,
            'type': 'FP32_Student_Noise',
            'path': distill_pth
        }

    # Student Quantized INT8
    int8_pth = os.path.join(checkpoints_dir, 'student_int8.pth')
    if os.path.exists(int8_pth):
        q_model = load_quantized_model(int8_pth, num_classes=10)
        models_to_test['Student (Quantized INT8)'] = {
            'model': q_model,
            'type': 'INT8_Quantized',
            'path': int8_pth
        }

    if not models_to_test:
        raise RuntimeError("No models found in checkpoints directory! Please run training first.")

    # 2. Measure CPU Latency & File Footprint for each model
    print("\n--- Profiling Onboard Edge Constraints (CPU Latency & Footprint) ---")
    latency_cache = {}
    footprint_cache = {}
    for name, info in models_to_test.items():
        size_mb = get_model_size_mb(info['path'])
        lat_info = measure_cpu_latency_ms(info['model'], input_shape=(1, 3, 64, 64), num_warmup=25, num_iter=100)
        latency_cache[name] = lat_info['mean_latency_ms']
        footprint_cache[name] = size_mb
        print(f"[{name}] Footprint: {size_mb:.3f} MB | CPU Latency: {lat_info['mean_latency_ms']:.2f} ms (p95: {lat_info['p95_latency_ms']:.2f} ms)")

    # 3. Define Stress Test Matrix
    # Gradual noise severity levels:
    test_conditions = [
        # Clean baseline
        ('Clean', 0.0, 'Clean (Baseline)'),
        # Gaussian Noise levels (std dev)
        ('Gaussian', 0.05, 'Gaussian σ=0.05'),
        ('Gaussian', 0.10, 'Gaussian σ=0.10'),
        ('Gaussian', 0.20, 'Gaussian σ=0.20'),
        ('Gaussian', 0.30, 'Gaussian σ=0.30'),
        ('Gaussian', 0.40, 'Gaussian σ=0.40'),
        # Salt & Pepper Pixel Defects (fraction)
        ('Salt & Pepper', 0.02, 'S&P 2%'),
        ('Salt & Pepper', 0.05, 'S&P 5%'),
        ('Salt & Pepper', 0.10, 'S&P 10%'),
        ('Salt & Pepper', 0.15, 'S&P 15%'),
        ('Salt & Pepper', 0.20, 'S&P 20%'),
        # Band Dropout (number of bands lost)
        ('Band Dropout', 1.0, 'Band Dropout (1 band)'),
        ('Band Dropout', 2.0, 'Band Dropout (2 bands)'),
    ]

    results_records = []
    best_cm = None

    print("\n--- Executing Headless Noise Robustness Evaluation Loop ---")
    for n_type, n_param, condition_label in test_conditions:
        evaluator = FixedNoiseEvaluator(noise_type=n_type, param=n_param)

        # Pre-generate noisy normalized batches for test set
        test_batches = []
        for raw_imgs, labels in test_loader:
            norm_imgs = evaluator.apply(raw_imgs, test_ds.norm_transform)
            test_batches.append((norm_imgs, labels))

        for model_name, info in models_to_test.items():
            model = info['model']
            all_preds, all_targets = [], []

            with torch.no_grad():
                for imgs, labels in test_batches:
                    outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1).numpy()
                    all_preds.extend(preds)
                    all_targets.extend(labels.numpy())

            metrics = compute_metrics(all_targets, all_preds)

            results_records.append({
                'model_name': model_name,
                'model_type': info['type'],
                'footprint_mb': footprint_cache[model_name],
                'cpu_latency_ms': latency_cache[model_name],
                'condition': condition_label,
                'noise_type': n_type,
                'noise_param': n_param,
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })

            # Save confusion matrix for quantized INT8 model under moderate Gaussian noise (σ=0.20)
            if 'INT8' in model_name and n_type == 'Gaussian' and np.isclose(n_param, 0.20):
                best_cm = metrics['confusion_matrix']

    # 4. Save results to CSV
    csv_path = os.path.join(deliverables_dir, 'benchmark_results.csv')
    df = pd.DataFrame(results_records)
    df.to_csv(csv_path, index=False)
    print(f"\n[Evaluate] Benchmark metrics written to {csv_path}")

    # 5. Generate plots silently
    print("[Evaluate] Generating publication-quality deliverable plots silently...")
    plot_f1_vs_noise(csv_path, os.path.join(deliverables_dir, 'f1_vs_noise_severity.png'))

    # Summary data for Pareto trade-off plot (evaluating under average sensor corruption)
    df_severe = df[df['condition'].isin(['Gaussian σ=0.20', 'S&P 10%', 'Band Dropout (1 band)'])]
    summary_data = []
    for m_name in models_to_test.keys():
        m_subset = df_severe[df_severe['model_name'] == m_name]
        avg_f1 = m_subset['macro_f1'].mean()
        summary_data.append({
            'name': m_name,
            'latency_ms': latency_cache[m_name],
            'f1_score': avg_f1,
            'size_mb': footprint_cache[m_name]
        })

    plot_latency_accuracy_tradeoff(summary_data, os.path.join(deliverables_dir, 'latency_vs_accuracy_tradeoff.png'))

    if best_cm is not None:
        plot_confusion_matrix(best_cm, EUROSAT_CLASSES, os.path.join(deliverables_dir, 'confusion_matrix.png'))

    # 6. Print Markdown Benchmark Table to Terminal
    print("\n")
    print("## 🛰️ Onboard Edge Benchmark Summary")
    print("")
    print("| Model Architecture | Size (MB) | Latency (ms) | Clean F1 | Noisy F1 |")
    print("|:---|---:|---:|---:|---:|")

    for item in summary_data:
        m_name = item['name']
        clean_row = df[(df['model_name'] == m_name) & (df['condition'] == 'Clean (Baseline)')].iloc[0]
        size_str = f"{item['size_mb']:.3f}"
        lat_str = f"{item['latency_ms']:.2f}"
        clean_f1_str = f"{clean_row['macro_f1']:.4f}"
        noisy_f1_str = f"{item['f1_score']:.4f}"
        print(f"| {m_name} | {size_str} | {lat_str} | {clean_f1_str} | {noisy_f1_str} |")

    print("")
    print("> **Noisy F1** = mean Macro F1 over Gaussian σ=0.20, S&P 10%, Band Dropout (1 band)")
    print("")
    print("### Deliverables generated in `./deliverables/`:")
    print("- `benchmark_results.csv`")
    print("- `f1_vs_noise_severity.png`")
    print("- `latency_vs_accuracy_tradeoff.png`")
    print("- `confusion_matrix.png`")
    print("- `sample_augmentations.png`")
    print("")


if __name__ == '__main__':
    run_evaluation()
