# WaveLetFusion

WaveLetFusion is a PyTorch-based project for cross-modal infrared-visible image registration and fusion.

This repository provides:

- a two-stage training pipeline
- a testing and evaluation pipeline
- a wavelet-based dense registration module
- a fusion network with cross-modal feature interaction
- visualization utilities for flow, deformation grids, registration results, and difference maps

---

## Highlights

- Image fusion
- Bidirectional registration supervision
- Wavelet-guided coarse-to-fine dense matching
- Cross-modal feature interaction with Swin-style window attention
- Rich visualization outputs for analysis and debugging

---

## Project Structure

```text
.
├── data
│   ├── dataset.py
│   ├── roadscene_train
│   │   ├── ir
│   │   ├── vi
│   │   ├── ir_d
│   │   ├── vi_d
│   │   ├── ir_flows
│   │   ├── vi_flows
│   │   ├── ir_valid
│   │   └── vi_valid
│   └── ROAD
│       ├── ir
│       └── vi
├── network
│   ├── network.py
│   └── loss.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
