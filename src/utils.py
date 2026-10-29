import os
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def get_device():
    """Returns optimal device for training (MPS for Mac Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def measure_cpu_latency_ms(model, input_shape=(1, 3, 64, 64), num_warmup=25, num_iter=100):
    """
    Measures exact CPU inference latency per image in milliseconds using time.perf_counter().
    Forces model to CPU to simulate edge satellite processor.
    """
    model_cpu = model.to('cpu').eval()
    dummy_input = torch.randn(*input_shape, device='cpu')

    # Warm-up passes
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cpu(dummy_input)

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_iter):
            t_start = time.perf_counter()
            _ = model_cpu(dummy_input)
            t_end = time.perf_counter()
            timings.append((t_end - t_start) * 1000.0)  # ms

    timings = np.array(timings)
    mean_lat = float(np.mean(timings))
    std_lat = float(np.std(timings))
    p95_lat = float(np.percentile(timings, 95))
    return {
        'mean_latency_ms': mean_lat,
        'std_latency_ms': std_lat,
        'p95_latency_ms': p95_lat
    }


def compute_metrics(y_true, y_pred):
    """
    Computes macro F1-score, weighted F1-score, accuracy, precision, and recall.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm
    }
