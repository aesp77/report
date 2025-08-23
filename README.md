# Research Project

This repository contains experiments and scripts for analyzing and modeling options volatility surfaces. The main goal is to build datasets of historical implied volatility, train generative models (VAE and LSTM), and calibrate stochastic volatility models such as Heston.

## Directory Structure

```
Project/
├── data/                     # Contains pre-generated datasets and intermediate results
│   ├── SPX_Index_history_dataset.csv
│   ├── vol_tensor_dataset.csv
│   └── ...
├── notebooks/                # Jupyter notebooks for experiments and analysis
│   ├── dataset_creator.ipynb
│   ├── Project_VAE_LTSM_clean.ipynb
│   ├── project_calibration.ipynb
│   ├── ...
├── src/                      # Source code for models and utilities
│   ├── data/                 # Data processing scripts
│   │   ├── loader.py         # upload dataset
│   │   ├── parametrization.py# runs heston and ssvi
│   │   ├── tensor_builder.py
│   │   ├── ssvi.py           # SSVI surface fitting
│   │   └── heston.py         # Heston model calibration
│   ├── models/               # Model definitions
│   │   ├── vae.py
│   │   ├── lstm.py
│   │   ├── decoder.py
│   │   └── no_arb_sl.py
│   ├── utils/                # Utility scripts
│   │   ├── plotting.py
│   │   ├── eval.py
│   │   └── simulation.py
│   └── ...
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── ...
```

## Dataset Requirements

The notebooks expect pre-generated CSV files in the `data/` directory:

* `SPX_Index_history_dataset.csv`
* `vol_tensor_dataset.csv`

These files can be produced using the notebook `notebooks/dataset_creator.ipynb`, which pulls market data from internal data sources. The scripts assume that these CSVs are available before running any training or calibration notebooks.

## Setup

1. Use **Python 3.10.11** (or newer).
2. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Latent Surface Modeling Pipeline

### 1. Problem Overview

At each timestep $t$, we observe:

* Volatility surface $y_t \in \mathbb{R}^{M \times K}$, where $M$ = maturities and $K$ = strikes.
* Market features $x_t \in \mathbb{R}^D$, such as yield curves, returns, volatilities, macro signals.

We define engineered features:

$f_t = \text{FeatureEngineer}(x_t, y_t) \in \mathbb{R}^{d_f}$

These are combined into:

$\tilde{x}_t = \text{flatten}(y_t) \Vert f_t \in \mathbb{R}^{M \cdot K + d_f}$

The goal is to encode $\tilde{x}_t$ into a latent vector $z_t$, model its dynamics, and decode it into forecasted IV surfaces.

### 2. Pipeline Architecture

#### Phase A — Latent Encoding

* PCA (baseline): linear compression.
* Autoencoder (AE): $\tilde{x}_t \rightarrow z_t$
* Variational AE (VAE): $\tilde{x}_t \rightarrow (\mu_t, \log \sigma_t^2), z_t \sim \mathcal{N}(\mu_t, \sigma_t^2)$

Trained to reconstruct $\tilde{x}_t$ with optional penalties (smoothness, no-arbitrage).

#### Phase B — Latent Temporal Forecasting

Given historical encoded latent vectors and features:

$[z_{t-L}, \dots, z_t] \oplus f_t \rightarrow \hat{z}_{t+1}$

Models:

* LSTM, GRU, Transformer (deep learning)
* PCA-VAR (linear autoregression baseline)
* Feature conditioning optional

#### Phase C — Decoding IV Surface

Given $\hat{z}_{t+1}$, reconstruct:

$\hat{y}_{t+1} = \text{Decoder}(\hat{z}_{t+1}) \quad \text{or} \quad \text{Decoder}(\hat{z}_{t+1}, f_{t+1})$

Decoder families:

* **SimpleSurfaceDecoder**: MLP-based with calendar/smile penalties
* **CNNDecoder**: latent-to-grid decoding via Conv2D
* **SurfaceRLDecoder**: residual correction over entire surfaces
* **SliceRLDecoder**: residual smile-level correction per $\tau$, using $[IV, \tau]$ as input

This framework enables forecasting and refining forward-looking volatility surfaces from market state embeddings.
