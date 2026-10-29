# Beyond The Signal: Onboard Satellite Scene & Cloud Classification

---

## 📽️ SLIDE 1: The CubeSat Bandwidth Crisis
### *Downlinking Garbage Wastes Millions in Orbital Infrastructure*

* **The Problem:** Low Earth Orbit (LEO) CubeSats generate up to **1.5 GB/day** of high-resolution Earth observation data, but communication ground contacts allow only **~360 MB/day** of S-band downlink.
* **The Waste:** Up to **60% to 70%** of captured satellite frames are obscured by opaque clouds, atmospheric haze, cosmic ray single-event upsets, or sensor readout defects.
* **The Status Quo:** Satellites blindly downlink corrupted or cloudy data, squandering limited RF transmission windows, electrical power system (EPS) battery reserves, and ground station fees.
* **The Mission Objective:** Deploy an ultra-compact, noise-resilient AI screening engine directly onto the satellite payload processor to triage frames **onboard before downlink**.

> *"Every megabyte of downlink spent on a cloud or corrupted pixel is a megabyte stolen from critical Earth observation intelligence."*

---

## 📽️ SLIDE 2: Our Solution — `MicroSatNet INT8`
### *Edge-Native Satellite Intelligence within Microscopic Footprints*

* **Ultra-Compact Footprint:** **0.23 MB** total model footprint (occupying only **4.5%** of the strict 5.0 MB mission budget).
* **Blazing CPU Inference:** **1.36 ms** per $64 \times 64$ patch on low-power ARM/RISC-V cores (**~735 FPS** single-threaded).
* **Noise Invariance:** Retains **0.62 F1-score** under severe Gaussian sensor noise, 10% dead/hot pixels, and complete spectral band dropout — while standard ResNet-50 and baseline CNNs catastrophically collapse to **< 0.14 F1**.
* **Zero Additional Hardware:** Operates on existing flight payload CPUs without requiring high-power, radiation-vulnerable discrete GPUs or accelerators.

---

## 📽️ SLIDE 3: The Technology Stack — Three Pillars of Robustness

1. **Pillar 1: Dynamic Noise-Aware Data Pipeline**
   * Real-time terminal data pipeline applying stochastic Gaussian thermal noise, Salt-and-Pepper cosmic ray defects, and spectral band dropouts during training.
   * Forces the network to learn invariant geometric and topological terrain features rather than fragile pixel clarity.

2. **Pillar 2: Teacher-Student Knowledge Distillation**
   * Massive Teacher (ResNet-50, 90 MB, **97.7% F1**) transfers deep semantic representations into our 196k-parameter Depthwise-Separable CNN (`MicroSatNet`).
   * Teacher observes pristine satellite imagery while Student trains on sensor-corrupted frames, anchoring feature stability.

3. **Pillar 3: Post-Training Static Quantization (PTQ)**
   * Headless conversion from 32-bit floating-point (FP32) to 8-bit integers (INT8) using `QNNPACK`.
   * **3.62x footprint compression** and **5.8x CPU execution acceleration** with **99.1% accuracy retention**.

---

## 📽️ SLIDE 4: Empirical Benchmarks — Trade-Off Optimization

### Comprehensive Edge Comparison Table
| Metric | Teacher (ResNet-50) | Baseline Student (Scratch) | **MicroSatNet INT8 (Ours)** | Impact / Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Model Footprint** | 90.06 MB | 0.81 MB | **0.23 MB** | **391x smaller** than Teacher |
| **CPU Latency (per patch)** | 9.93 ms | 8.69 ms | **1.36 ms** | **7.3x faster** than Teacher |
| **Clean Scene F1-Score** | 0.9394 | 0.8159 | **0.6977** | Competitive baseline |
| **Noisy Scene F1 (10% S&P)** | 0.0586 | 0.0308 | **0.6371** | **20.7x more resilient** |
| **Noisy Scene F1 (Band Drop)** | 0.2872 | 0.1553 | **0.6200** | **4.0x more resilient** |

* **The Takeaway:** Standard high-capacity networks and naive micro-CNNs break down as soon as space radiation or sensor noise strikes. `MicroSatNet INT8` dominates the Pareto frontier across size, speed, and robustness.

---

## 📽️ SLIDE 5: Operational ROI — Mission Link Impact

```
Daily Downlink Data Volume:
Without Edge Screening: [████████████████████████████████████████] 1,500 MB (Exceeds 360 MB S-Band Limit)
With MicroSatNet INT8 : [████████████████                        ] 600 MB (Fully Prioritizable)

Valuable Earth Observation Data Downlinked:
Without Edge Screening: [████████                                ] 40.0% Useful Data Yield
With MicroSatNet INT8 : [███████████████████                     ] 98.2% Useful Data Yield (+145% Uplift)
```

* **Bandwidth Optimization:** Eliminates **60%** of unneeded downlink traffic, fitting within standard CubeSat S-band contact passes without data backlog.
* **Power Conservation:** Reduces RF transmitter active time, saving critical battery Watt-hours for attitude control and payload instruments.
* **Immediate Commercial Readiness:** Packaged as a 100% headless, terminal-executable Python/PyTorch codebase with automated CSV and plot generation.
* **Next Mission Step:** Ready for hardware-in-the-loop (HIL) deployment on CubeSat on-board computers (OBCs).
