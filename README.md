# Volatility Surface Modeling & Forecasting Project
This repository contains the full codebase, data, and experiments for modeling, forecasting, and analyzing financial options volatility surfaces.
The project implements a modular deep learning pipeline that combines machine learning, time series forecasting, and financial engineering.

## Project Overview
The objective is to encode historical implied volatility surfaces and market features into latent representations, model their temporal dynamics, and decode them to forecast future volatility surfaces.
Volatility surfaces reflect the underlying return distribution and interactions with broader market factors. This project develops a prototype forecasting framework that learns these interactions from both volatility surfaces and engineered market features 

## Project Highlights
- **End-to-end modular pipeline**: encoders - temporal models - decoders - evaluation.  
- **26+ full pipeline configurations** benchmarked systematically.  
- **Comparative framework**: models ranked jointly on statistical accuracy (RMSE), arbitrage consistency, and temporal stability.  

## Key Contributions
- A **Black–Scholes hybrid loss** weighting errors by option sensitivities (Vega, Gamma).  
- A **Mixture-of-Experts (MoE) decoder** with explicit maturity/moneyness specialisation, annealed biases, and diversity regularisation.  
- Demonstration of the common **“good RMSE, bad fit” pathology** and how structured models resolve it.  

## Reproducible Experiments

Two notebooks allow the examiner to reproduce the main results without retraining:

- *pipelines_load_models.ipynb*: loads pre-trained models, assembles full pipelines, and benchmarks performance against PCA-VAR baselines. Approximate runtime: 20 minutes with GPU.

- *final_pipeline.ipynb*: runs a complete end-to-end pipeline (VAE encoder, GRU temporal model, VAE decoder, MoE corrective decoder). Approximate runtime: 4 hours with GPU.

These notebooks are sufficient to demonstrate the system end-to-end and reproduce the comparative analysis.

*Note*: All models are implemented in Keras 3 API with PyTorch backend and are GPU-enabled.



## Directory Structure

```
report/
├── data/                  # Datasets and intermediate results
│   ├── SPX_Index_history_dataset.csv
│   ├── vol_tensor_dataset.csv
├── notebooks/             # Jupyter notebooks for experiments and analysis
│   ├── baseline.ipynb
│   ├── decoders_experiments_ae_gru.ipynb
│   ├── decoders_experiments_vae_gru.ipynb
│   ├── decoders_pointwise_ae.ipynb
│   ├── decoders_pointwise_fwpca.ipynb
│   ├── decoders_pointwise_vae.ipynb
│   ├── edav.ipynb
│   ├── encoders.ipynb
│   ├── final_pipeline.ipynb   # display 1 pipeline , run
│   ├── pipelines_load_models.ipynb #dispaly summary results for experiments 
│   ├── raw_dataset_generator.ipynb
│   ├── temporal.ipynb
│   ├── temporal_ae.ipynb
│   ├── temporal_fwpca.ipynb
│   ├── temporal_vae2.ipynb
│   ├── saved_images/      # Plots and figures
│   └── saved_models/      # Trained model files
├── src/                   # Source code for data, models, and utilities
│   ├── data/              # Data processing and feature engineering
│   │   ├── dataset.py
│   │   ├── dataset_builder.py
│   │   ├── edav.py
│   │   ├── feature_engineering.py
│   │   ├── loader.py
│   │   └── tensor_builder.py
│   ├── models/            # Model definitions and decoders
│   │   ├── ae.py
│   │   ├── cnn.py
│   │   ├── decoder.py
│   │   ├── decoder_losses.py
│   │   ├── encoder_pca.py
│   │   ├── forecaster_var.py
│   │   ├── fw_pca.py
│   │   ├── gbo.py
│   │   ├── gru.py
│   │   ├── lstm.py
│   │   ├── pointwise_moe.py
│   │   ├── pointwise_moe_adpt.py
│   │   ├── rl_decoder.py
│   │   ├── ssvi.py
│   │   ├── transformer.py
│   │   ├── vae_l1.py
│   │   ├── vae_l2.py
│   │   ├── vae_mle.py
│   │   ├── vae_mle_b.py
│   │   ├── vae_mle_fv.py
│   │   └── ...
│   ├── utils/             # Utility scripts for plotting, evaluation, routing
│   │   ├── eval.py
│   │   ├── moe_weights.py
│   │   ├── plot_error.py
│   │   ├── plotting.py
│   │   └── pointwise_router.py
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # License information
```

## Data Requirements

- Place the following CSV files in the `data/` directory:
  - `SPX_Index_history_dataset.csv`: Historical index and market data
  - `vol_tensor_dataset.csv`: Precomputed volatility surface tensors
- These can be generated using `notebooks/raw_dataset_generator.ipynb` or other data scripts.

## Notebooks

- All main experiments, model training, and analysis are performed in the `notebooks/` folder.
- Notebooks are organized by pipeline stage (encoding, temporal modeling, decoding, evaluation).
- Visualizations and saved model weights are stored in `notebooks/saved_images/` and `notebooks/saved_models/`.

## Source Code Modules

- `src/data/`: Data loading, feature engineering, and tensor construction
- `src/models/`: Model architectures (PCA, AE, VAE, LSTM, GRU, Transformer, MoE, CNN, decoders)
- `src/utils/`: Plotting, evaluation, region routing, and helper functions

## Pipeline Summary

1. **Encoding**: Compress volatility surfaces and features into latent vectors using PCA, AE, or VAE.
2. **Temporal Modeling**: Forecast latent vectors using LSTM, GRU, Transformer, or VAR models.
3. **Decoding**: Reconstruct future volatility surfaces from predicted latent vectors using various decoders (MLP, CNN, RL-based, MoE).
4. **Evaluation**: Assess model performance, visualize results, and analyze error regions.

## Setup Instructions

1. Create a Python virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place required data files in the `data/` directory.
3. Run experiments and analysis using the Jupyter notebooks in `notebooks/`.

## Model Families & Features

- **PCAEncoder, AE, VAE**: Latent encoding of surfaces and features
- **LSTM, GRU, Transformer**: Temporal forecasting of latent states
- **Decoders**: Corrective MLP, CNN, RL-based, MoE, pointwise and slice-level correction
- **Evaluation**: Error analysis, region-based routing, and visualization

## Outputs

- Forecasted volatility surfaces, error heatmaps, model weights, and plots
- All outputs are saved in `notebooks/saved_images/` and `notebooks/saved_models/`

## License

This project is licensed under the terms of the LICENSE file in the repository.

---


