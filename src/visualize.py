import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.dataset import EUROSAT_CLASSES


def plot_f1_vs_noise(csv_path, output_path='deliverables/f1_vs_noise_severity.png'):
    """
    Plots Classification F1-Score degradation as noise severity increases.
    Generates 3 subplots: Gaussian Noise, Salt & Pepper Noise, and Band Dropout.
    Strictly terminal-only without plt.show().
    """
    if not os.path.exists(csv_path):
        print(f"[Visualize] CSV not found at {csv_path}, skipping plot.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('#0d1117')

    palette = {
        'Teacher (ResNet-50 FP32)': '#f78166',
        'Student Baseline (Scratch FP32)': '#8b949e',
        'Student (Distilled+Noise FP32)': '#58a6ff',
        'Student (Quantized INT8)': '#3fb950'
    }
    markers = {
        'Teacher (ResNet-50 FP32)': 's',
        'Student Baseline (Scratch FP32)': 'x',
        'Student (Distilled+Noise FP32)': 'o',
        'Student (Quantized INT8)': '^'
    }

    # 1. Gaussian Noise
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    df_gauss = df[df['noise_type'] == 'Gaussian'].sort_values('noise_param')
    for model_name, grp in df_gauss.groupby('model_name'):
        color = palette.get(model_name, '#a371f7')
        marker = markers.get(model_name, 'o')
        ax1.plot(grp['noise_param'], grp['macro_f1'], label=model_name,
                 color=color, marker=marker, linewidth=2.2, markersize=7)
    ax1.set_title('Robustness: Gaussian Sensor Noise (σ)', color='white', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Noise Std Dev (σ)', color='#c9d1d9', fontsize=11)
    ax1.set_ylabel('Macro F1-Score', color='#c9d1d9', fontsize=11)
    ax1.tick_params(colors='#8b949e')
    ax1.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    ax1.set_ylim([0.0, 1.02])

    # 2. Salt & Pepper Noise
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    df_sp = df[df['noise_type'] == 'Salt & Pepper'].sort_values('noise_param')
    for model_name, grp in df_sp.groupby('model_name'):
        color = palette.get(model_name, '#a371f7')
        marker = markers.get(model_name, 'o')
        ax2.plot(grp['noise_param'] * 100, grp['macro_f1'], label=model_name,
                 color=color, marker=marker, linewidth=2.2, markersize=7)
    ax2.set_title('Robustness: Salt & Pepper Pixel Defects (%)', color='white', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Defective Pixels (%)', color='#c9d1d9', fontsize=11)
    ax2.set_ylabel('Macro F1-Score', color='#c9d1d9', fontsize=11)
    ax2.tick_params(colors='#8b949e')
    ax2.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    ax2.set_ylim([0.0, 1.02])

    # 3. Band Dropout
    ax3 = axes[2]
    ax3.set_facecolor('#161b22')
    df_band = df[df['noise_type'] == 'Band Dropout'].sort_values('noise_param')
    for model_name, grp in df_band.groupby('model_name'):
        color = palette.get(model_name, '#a371f7')
        marker = markers.get(model_name, 'o')
        ax3.plot(grp['noise_param'], grp['macro_f1'], label=model_name,
                 color=color, marker=marker, linewidth=2.2, markersize=7)
    ax3.set_title('Robustness: Band Dropout (Sensor Failure)', color='white', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Dropped Spectral Bands', color='#c9d1d9', fontsize=11)
    ax3.set_ylabel('Macro F1-Score', color='#c9d1d9', fontsize=11)
    ax3.tick_params(colors='#8b949e')
    ax3.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    ax3.set_ylim([0.0, 1.02])
    ax3.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', loc='lower left', fontsize=9)

    fig.suptitle('Classification F1-Score vs. Synthetic Sensor Noise Severity (CubeSat Stress Test)',
                 color='white', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualize] F1 vs Noise plot successfully written to {output_path}")


def plot_latency_accuracy_tradeoff(summary_data, output_path='deliverables/latency_vs_accuracy_tradeoff.png'):
    """
    Scatter/Bubble plot: CPU Inference Latency (ms) vs Macro F1 Score under Noise.
    Bubble diameter corresponds to model size (MB).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    colors = ['#f78166', '#8b949e', '#58a6ff', '#3fb950']

    for i, item in enumerate(summary_data):
        name = item['name']
        lat = item['latency_ms']
        f1 = item['f1_score']
        size = item['size_mb']
        color = colors[i % len(colors)]

        # Bubble size proportional to model size (log-scaled for visibility)
        bubble_size = max(120, int(np.sqrt(size) * 120))

        ax.scatter(lat, f1, s=bubble_size, color=color, alpha=0.85, edgecolors='white', linewidth=1.5, zorder=5)

        # Annotation
        ax.annotate(
            f"{name}\n({size:.2f} MB | {lat:.2f} ms)",
            xy=(lat, f1),
            xytext=(10, -15 if i % 2 == 0 else 10),
            textcoords='offset points',
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=color, alpha=0.9),
            zorder=6
        )

    ax.set_title('Onboard Edge Trade-Off: CPU Latency vs. F1-Score (Bubble Size = Footprint MB)',
                 color='white', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('CPU Inference Latency per Image (ms) [Lower is Better]', color='#c9d1d9', fontsize=11)
    ax.set_ylabel('Robust Test Macro F1-Score [Higher is Better]', color='#c9d1d9', fontsize=11)
    ax.tick_params(colors='#8b949e')
    ax.grid(True, linestyle='--', alpha=0.3, color='#30363d')

    # Draw ideal target quadrant
    ax.axvspan(0, 15, alpha=0.08, color='green', label='Target Edge Constraint Zone (<15 ms)')

    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualize] Latency vs Accuracy trade-off plot written to {output_path}")


def plot_confusion_matrix(cm, class_names=EUROSAT_CLASSES, output_path='deliverables/confusion_matrix.png'):
    """
    Plots normalized confusion matrix heatmaps silently.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#8b949e')

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Quantized INT8 MicroSatNet: Confusion Matrix under Sensor Noise',
        ylabel='True Satellite Land Cover',
        xlabel='Predicted Land Cover'
    )
    ax.title.set_color('white')
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.xaxis.label.set_color('#c9d1d9')
    ax.tick_params(colors='#8b949e')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] < thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualize] Confusion matrix plot written to {output_path}")
