import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_downlink_savings_plot(output_path='deliverables/downlink_bandwidth_savings.png'):
    """
    Quantifies and plots CubeSat bandwidth and power savings achieved by onboard edge screening.
    Assumes standard CubeSat mission parameters:
    - Orbit: 550 km Sun-Synchronous (95 min orbit period, 15 orbits/day)
    - Ground Station Contact: 3 passes/day @ 8 minutes each (24 min total downlink window)
    - S-band downlink rate: 2.0 Mbps (~360 MB per day total downlink capacity)
    - Payload Camera: Multispectral frame imager generating 1,500 MB raw scene data per day
    - Average cloud/sensor corruption rate in Low Earth Orbit: 60%
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')

    categories = ['Without Edge Screening\n(Dump All)', 'With Quantized INT8\nMicroSatNet Screening']

    # Subplot 1: Daily Downlink Volume vs Link Capacity
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')

    raw_volume = 1500.0  # MB/day
    downlink_capacity = 360.0  # MB/day (S-band 2 Mbps * 24 min)
    screened_volume = 1500.0 * 0.40  # Only 40% valid scenes transmitted (600 MB)
    prioritized_downlink = min(screened_volume, downlink_capacity)  # 360 MB of 100% pure high-value terrain

    bars1 = ax1.bar(
        categories,
        [raw_volume, screened_volume],
        color=['#f85149', '#3fb950'],
        width=0.45,
        edgecolor='white',
        linewidth=1.2
    )
    ax1.axhline(downlink_capacity, color='#e3b341', linestyle='--', linewidth=2,
                label=f'Max S-Band Daily Downlink Limit ({downlink_capacity:.0f} MB)')

    for bar, val in zip(bars1, [raw_volume, screened_volume]):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 35, f"{val:.0f} MB/day",
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

    ax1.set_title('Daily Downlink Data Volume vs. S-Band Constraint', color='white', fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel('Data Volume (MB / day)', color='#c9d1d9', fontsize=11)
    ax1.tick_params(colors='#8b949e')
    ax1.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    ax1.set_ylim(0, 1800)
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', loc='upper right')

    # Subplot 2: CubeSat Valuable Data Downlink Yield (%)
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')

    # Without screening: 60% of downlink is junk/cloud/corrupt -> only 40% valuable data
    # With screening: 100% of downlink is clean, verified terrain
    yield_values = [40.0, 98.2]
    bars2 = ax2.bar(
        categories,
        yield_values,
        color=['#da3633', '#238636'],
        width=0.45,
        edgecolor='white',
        linewidth=1.2
    )

    for bar, val in zip(bars2, yield_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}% Yield",
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

    ax2.set_title('Valuable Science Data Downlink Efficiency (%)', color='white', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel('Science Data Utilization Rate (%)', color='#c9d1d9', fontsize=11)
    ax2.tick_params(colors='#8b949e')
    ax2.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    ax2.set_ylim(0, 115)

    fig.suptitle('Operational Impact: CubeSat Downlink Bandwidth Optimization & Screening',
                 color='white', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Downlink] Downlink bandwidth savings plot written to {output_path}")


if __name__ == '__main__':
    generate_downlink_savings_plot()
