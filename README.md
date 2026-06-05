# PUF-Chain NDS

A research simulation and reproducibility framework for a hardware-rooted IoT security system that combines Physically Unclonable Functions (PUFs), blockchain-style tamper detection, adaptive error correction, and machine-learning-based intrusion detection.

https://doi.org/10.5281/zenodo.20554095

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [How It Works](#how-it-works)
  - [PUF Hardware Models](#1-puf-hardware-models)
  - [The PUF-Chain Block](#2-the-puf-chain-block)
  - [Sensors and the Network Monitor](#3-sensors-and-the-network-monitor)
  - [Error Correction — Fuzzy Extractor](#4-error-correction--fuzzy-extractor)
- [Experiments](#experiments)
  - [Dataset Generation](#dataset-generation)
  - [v2 — Adaptive ECC and Intrusion Detection](#v2--adaptive-ecc-and-intrusion-detection)
  - [v1.1 — Robustness and Consensus](#v11--robustness-and-consensus)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Reproducing the Paper](#reproducing-the-paper)

---

## Overview

**PUF-Chain NDS** breaks down into three ideas:

| Term | Meaning |
|------|---------|
| **PUF** | *Physically Unclonable Function* — a hardware fingerprint born from manufacturing imperfections |
| **Chain** | A blockchain-inspired ledger that links each sensor transmission cryptographically |
| **NDS** | *Network Detection System* — an anomaly detector that watches for attackers |

Together, they form a system where IoT sensors prove their identity through physics rather than stored keys, chain their transmissions for tamper-evidence, and feed a network monitor that learns to distinguish normal noise from adversarial behavior.

---

## The Problem

Imagine thousands of IoT sensors deployed in the field — weather stations, industrial monitors, smart meters. Each one must prove it is a legitimate, untampered device, and every reading it sends must be verifiable.

Traditional approaches store a secret cryptographic key in flash memory, but a key can be extracted by an adversary with physical access. A PUF sidesteps this: instead of *storing* a secret, the device *is* the secret. Its response to a given challenge is determined by tiny, random variations in transistor delays and cell startup voltages introduced during fabrication — variations that cannot be perfectly cloned even by the original manufacturer.

The complication is that PUF responses are not perfectly stable. Heat, voltage fluctuation, and aging all introduce bit-flip noise. This project models that noise, corrects for it adaptively, and then uses the very noise patterns as a signal to detect when something is wrong.

---

## How It Works

### 1. PUF Hardware Models

`puf_models.py` simulates three classes of PUF hardware, each calibrated to published characterization data:

| PUF Type | Intra-HD (noise) | Inter-HD (uniqueness) | Character |
|----------|------------------|-----------------------|-----------|
| **SRAM** | 3.20% | 46.89% | Moderate noise, good uniqueness |
| **Arbiter** | 0.47% | 45.15% | Very stable under normal conditions |
| **Hybrid** | 6.71% | 50.30% | Noisiest, but maximally unique |

- **Intra-HD** measures how much a device's own response varies across repeated reads. Lower is more stable.
- **Inter-HD** measures how different two devices look from each other. Closer to 50% is ideal — like two sides of a fair coin.

Each model computes per-bit flip probabilities as a function of temperature deviation, voltage variation, and device age:

```
p_flip(bit) = b + α·|T − 25°C| + β·|V − 1.8V| + γ·(age / 365)
```

The four coefficients (`b`, `α`, `β`, `γ`) are drawn randomly per bit at construction time, giving each simulated device a unique noise profile.

---

### 2. The PUF-Chain Block

`puf_chain.py` defines how a sensor wraps its reading into a tamper-evident block:

1. **Encrypt** — the JSON payload is XOR'd against a keystream derived from the device's noisy PUF response.
2. **Hash** — the ciphertext, previous block hash, and counter are fed into SHA-256, chaining blocks like a miniature blockchain.
3. **Package** — the result is a block containing the counter, previous hash, current hash, and base64-encoded ciphertext.

If an attacker tampers with any past transmission, the hash chain breaks. Replay attacks and message substitution become detectable by any party that holds the reference chain.

```
block_hash = SHA256( prev_hash | counter | XOR(payload, puf_keystream) )
```

---

### 3. Sensors and the Network Monitor

`devices_and_nds.py` provides two classes:

**`Sensor`** — represents a physical IoT device. On each transmission it:
- Queries its PUF for a clean reference response and a noisy environmental response
- Computes the **flip mask** (which bits changed) and embeds it in the block
- Increments its chain counter and returns the complete block

**`NDS`** (Network Detection System) — the gateway-side monitor. It receives blocks and logs a structured row per packet:

```
timestamp, device_id, puf_type, temp, voltage, age_days, is_attack,
counter, n_flipped, flipped_0 … flipped_63
```

This 73-column log is the training and evaluation dataset for all downstream ML experiments.

---

### 4. Error Correction — Fuzzy Extractor

`fuzzy_extractor.py` handles the fact that PUF responses are noisy, which makes raw key derivation unreliable.

The solution is **repetition coding**: each bit is voted on across multiple reads. If 5 out of 7 reads agree on `1`, the output is `1`. The module computes the exact number of repetitions needed per bit to hit a target key-recovery probability, enabling *adaptive* sizing:

- Stable bits (low flip probability) need only 1–3 repetitions.
- Noisy bits need up to 9.
- This beats a fixed rep-8 scheme in both efficiency and reliability.

---

## Experiments

### Dataset Generation

`simulate.py` generates **18,000 sensor transmissions** — 6,000 per PUF family — across four environmental regimes:

| Regime | Temperature | Voltage Jitter | Label |
|--------|-------------|----------------|-------|
| Cold | 10–25°C | ±0.03V | benign |
| Nominal | 25–45°C | ±0.02V | benign |
| Warm | 45–65°C | ±0.03V | benign |
| Attack | 65–90°C | ±0.10V + ±0.20V spike | **attack** |

For SRAM, the split is: 2,676 cold / 2,552 nominal / 302 warm / 470 attack.
Arbiter and Hybrid use: 5,530 nominal / 470 attack each.

Output CSVs land in `dataset/` and metrics are written to `output/results.json`.

---

### v2 — Adaptive ECC and Intrusion Detection

`run_experiments_v2.py` runs two parallel analyses on the SRAM dataset.

**Predicting flip rate for adaptive ECC sizing:**

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost (aggregate) | MAE | 1.26 bits |
| XGBoost (aggregate) | R² | 0.363 |
| Random Forest (per-bit) | Mean RMSE | 0.0625 |
| Random Forest (per-bit) | Mean R² | −0.50 |

Using RF-predicted per-bit flip probabilities, the adaptive ECC allocates:

| Strategy | Cells used | Key recovery (nominal) |
|----------|-----------|------------------------|
| Static rep-8 | 512 | 99.24% |
| **Adaptive** | **420** | **99.64%** |
| Static rep-8 under attack | 512 | 68.29% |

Adaptive ECC achieves an **18% cell saving** while *improving* key recovery at nominal conditions, and the per-regime analysis clearly exposes why a static scheme fails under attack.

**Intrusion detection with Isolation Forest:**

Trained on 4,424 benign packets, evaluated on 1,106 benign + 470 adversarial:

| Metric | Value |
|--------|-------|
| Recall | **95.7%** |
| False alarm rate | 5.7% |
| Precision | 87.7% |
| ROC-AUC | **0.9939** |

---

### v1.1 — Robustness and Consensus

`run_experiments_v11.py` covers three advanced topics:

**Federated Learning Robustness (Trimmed Mean vs. FedAvg)**

When aggregating flip-rate models across multiple gateway nodes, adversarial nodes can poison a standard FedAvg round. Trimmed-mean aggregation discards outlier updates before averaging:

| Aggregation | Attack-regime bit MAE |
|-------------|----------------------|
| FedAvg | 0.0414 |
| **Trimmed Mean** | **0.0022** |
| Improvement ratio | **18.8×** |

**Arrhenius Cold-Start Correction**

A physics-based Arrhenius model for low-temperature PUF behavior closes **23.9%** of the prediction gap that a naive baseline leaves open, improving key enrollment reliability during cold-boot scenarios.

**PBFT Consensus Overhead**

Byzantine Fault Tolerant consensus among gateway nodes has message complexity `(n−1) + 2n(n−1) + (f+1)`:

| Cluster | Faults tolerated | Messages |
|---------|-----------------|---------|
| n = 4 | f = 1 | 29 |
| n = 7 | f = 2 | 93 |

---

## Key Results

| Headline | Value |
|----------|-------|
| Adaptive ECC cells (vs. 512 static) | **420 — 18% saving** |
| Adaptive key recovery probability | **99.64%** |
| Isolation Forest ROC-AUC | **0.9939** |
| Isolation Forest recall on attack traffic | **95.7%** |
| Trimmed-mean improvement over FedAvg | **18.8×** |
| Arrhenius cold-start gap closed | **23.9%** |

---

## Project Structure

```
puf-chain-nds/
│
├── puf_models.py               # SRAM, Arbiter, Hybrid PUF simulation
├── puf_chain.py                # Blockchain-style block construction & hashing
├── devices_and_nds.py          # Sensor and NDS gateway classes
├── fuzzy_extractor.py          # Repetition-code error correction math
│
├── simulate.py                 # Generates the 18k-row labeled dataset
├── run_experiments_v2.py       # Adaptive ECC + Isolation Forest experiments
├── run_experiments_v11.py      # Federated robustness + Arrhenius + PBFT
├── make_figures.py             # Writes figure manifest from results.json
├── execute_notebook.py         # Notebook execution helper
│
├── puf_chain_nds_kaggle.ipynb  # End-to-end reproducibility notebook
│
├── dataset/
│   ├── traffic_sram.csv        # 6,000 SRAM sensor packets
│   ├── traffic_arbiter.csv     # 6,000 Arbiter sensor packets
│   └── traffic_hybrid.csv      # 6,000 Hybrid sensor packets
│
└── output/
    ├── results.json            # All computed metrics (seed-locked)
    └── figure_manifest.json    # Highlight values for publication figures
```

---

## Getting Started

**Requirements:** Python 3.10+, NumPy

```bash
# Install dependencies
pip install numpy

# 1. Generate the dataset
python simulate.py

# 2. Run the experiments
python run_experiments_v2.py
python run_experiments_v11.py

# 3. Build the figure manifest
python make_figures.py
```

All outputs are written to `dataset/` and `output/`. The `results.json` file accumulates metrics from each script and is safe to re-run — scripts merge their section in rather than overwriting.

---

## Reproducing the Paper

The Jupyter notebook `puf_chain_nds_kaggle.ipynb` is the single entry point for full reproducibility. It:

1. Runs the entire pipeline (`simulate` → `run_experiments_v2` → `run_experiments_v11`)
2. Loads `output/results.json`
3. Asserts every headline metric against paper-reported tolerances

Open it from the repository root and run all cells. The final cell prints `All paper-aligned checks passed.` if reproduction is successful.

All randomness is seeded at `seed = 42`. Results are deterministic across platforms given the same NumPy version.
