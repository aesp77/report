
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# list of functions for evaluating forecasts
# These functions can be used to assess the performance of different forecasting models.

def evaluate_forecast(X_true, X_decoded):
    """
    Compute MSE and MAE between true and decoded arrays.
    Args:
        X_true (ndarray): Ground truth values
        X_decoded (ndarray): Decoded/predicted values
    Returns:
        mse, mae, X_true, X_decoded
    """
    n = min(X_true.shape[0], X_decoded.shape[0])
    X_true, X_decoded = X_true[:n], X_decoded[:n]
    mse = mean_squared_error(X_true, X_decoded)
    mae = mean_absolute_error(X_true, X_decoded)
    return mse, mae, X_true, X_decoded


def print_evaluation(mse, mae, label="Forecast"):
    """
    Print formatted MSE and MAE for a forecast.
    """
    print(f"{label} MSE: {mse:.6f} | MAE: {mae:.6f}")



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns


def evaluate_pca_var(
    pca, Z_train, Z_test, Z_pred, n_components=None,
    plot_time_series=True, plot_residuals=True
):
    """
    Plot PCA diagnostics and component-wise forecast evaluation.
    Args:
        pca: fitted PCA object
        Z_train (ndarray): latent vectors from training data
        Z_test (ndarray): true latent vectors from test data
        Z_pred (ndarray): predicted latent vectors from VAR
        n_components (int): number of PCA components to use/display
        plot_time_series (bool): show per-PC forecast vs true
        plot_residuals (bool): show latent residual heatmap
    """
    if n_components is None:
        n_components = Z_pred.shape[1]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # scree plot
    axs[0, 0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    axs[0, 0].set_xlabel("Number of PCA components")
    axs[0, 0].set_ylabel("Cumulative explained variance")
    axs[0, 0].set_title("PCA Scree Plot")
    axs[0, 0].grid(True)

    # latent space: training vs test
    axs[0, 1].scatter(Z_train[:, 0], Z_train[:, 1], label="Train", alpha=0.6)
    axs[0, 1].scatter(Z_test[:, 0], Z_test[:, 1], label="Test", alpha=0.6)
    axs[0, 1].set_title("PCA Latent Space")
    axs[0, 1].set_xlabel("PC1")
    axs[0, 1].set_ylabel("PC2")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # forecast vs true in PC space (first 2 components)
    axs[1, 0].scatter(Z_test[:len(Z_pred), 0], Z_test[:len(Z_pred), 1], label="True", alpha=0.5)
    axs[1, 0].scatter(Z_pred[:, 0], Z_pred[:, 1], label="Forecast", alpha=0.5)
    axs[1, 0].set_title("Forecast vs True Trajectories (PC1-PC2)")
    axs[1, 0].set_xlabel("PC1")
    axs[1, 0].set_ylabel("PC2")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # mse per component
    mse_vals = [mean_squared_error(Z_test[:len(Z_pred), i], Z_pred[:, i]) for i in range(n_components)]
    axs[1, 1].bar([f"PC{i+1}" for i in range(n_components)], mse_vals)
    axs[1, 1].set_title("MSE per PCA Component")
    axs[1, 1].set_ylabel("MSE")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # time series plots per component
    if plot_time_series:
        fig, axs = plt.subplots(n_components, 1, figsize=(12, 2.5 * n_components), sharex=True)
        for i in range(n_components):
            axs[i].plot(Z_test[:len(Z_pred), i], label="True", linestyle="--")
            axs[i].plot(Z_pred[:, i], label="Forecast", alpha=0.7)
            axs[i].set_title(f"PC{i+1}")
            axs[i].grid(True)
        axs[0].legend()
        plt.suptitle("Latent Forecast vs True per Component")
        plt.tight_layout()
        plt.show()

    # heatmap of residuals
    if plot_residuals:
        residuals = Z_test[:len(Z_pred)] - Z_pred
        plt.figure(figsize=(12, 4))
        sns.heatmap(residuals.T, cmap="coolwarm", center=0, cbar_kws={"label": "Residual"})
        plt.xlabel("Forecast Step")
        plt.ylabel("PCA Component")
        plt.title("Residual Heatmap: Z_true - Z_pred")
        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_pca_var2(pca, Z_train, Z_test, Z_pred, n_components=None):
    """
    Full diagnostic plot: Scree, PC1/2 trajectory, MSE bar, residual heatmap, time series.
    """
    if n_components is None:
        n_components = Z_pred.shape[1]

    # layout setup
    fig = plt.figure(constrained_layout=True, figsize=(10, 2 * (2 + n_components)))
    subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, n_components])
    axs_top = subfigs[0].subplots(1, 2)
    axs_mid = subfigs[1].subplots(1, 2)
    axs_bottom = subfigs[2].subplots(n_components, 1, sharex=True)

    # top row: variance and trajectory
    axs_top[0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o", lw=1.5)
    axs_top[0].set_title("Cumulative Explained Variance")
    axs_top[0].set_xlabel("PC")
    axs_top[0].set_ylabel("Cumulative Variance")
    axs_top[0].grid(True)

    axs_top[1].scatter(Z_test[:len(Z_pred), 0], Z_test[:len(Z_pred), 1], label="True", s=10, alpha=0.6)
    axs_top[1].scatter(Z_pred[:, 0], Z_pred[:, 1], label="Forecast", s=10, alpha=0.6)
    axs_top[1].set_title("Forecast vs True (PC1–PC2)")
    axs_top[1].set_xlabel("PC1")
    axs_top[1].set_ylabel("PC2")
    axs_top[1].legend()
    axs_top[1].grid(True)

    # middle row: mse and residuals
    mse_vals = [mean_squared_error(Z_test[:len(Z_pred), i], Z_pred[:, i]) for i in range(n_components)]
    axs_mid[0].bar(range(1, n_components + 1), mse_vals)
    axs_mid[0].set_title("MSE per PCA Component")
    axs_mid[0].set_xlabel("PC")
    axs_mid[0].set_ylabel("MSE")
    axs_mid[0].grid(True)

    residuals = Z_test[:len(Z_pred)] - Z_pred
    sns.heatmap(residuals.T, ax=axs_mid[1], cmap="coolwarm", center=0, cbar=False)
    axs_mid[1].set_title("Residual Heatmap")
    axs_mid[1].set_xlabel("Step")
    axs_mid[1].set_ylabel("PC")

    # bottom: time series per PC
    for i in range(n_components):
        axs_bottom[i].plot(Z_test[:len(Z_pred), i], label="True", linestyle="--")
        axs_bottom[i].plot(Z_pred[:, i], label="Forecast", alpha=0.7)
        axs_bottom[i].set_ylabel(f"PC{i+1}")
        axs_bottom[i].grid(True)
    axs_bottom[0].legend()
    axs_bottom[-1].set_xlabel("Step")

    subfigs[2].suptitle("Latent Forecast vs True per Component", fontsize=14)
    plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate_encoder(encoder, X_surface, X_features, M, K, return_latent=False):
    """
    Evaluate encoder + decoder reconstruction from combined [surface, features].
    Args:
        encoder: object with .fit_transform() and .inverse_transform()
        X_surface: np.ndarray, shape (T, M*K)
        X_features: np.ndarray, shape (T, D)
        M, K: grid shape (maturities, strikes)
        return_latent: if True, returns Z in addition to RMSE
    Returns:
        rmse_surface, X_recon_surface (T, M, K), [optional Z]
    """
    X_combined = np.hstack([X_surface, X_features])

    # filter valid rows
    valid_mask = np.all(np.isfinite(X_combined), axis=1)
    X_combined_valid = X_combined[valid_mask]
    X_surface_valid = X_surface[valid_mask]

    # fit + decode
    Z = encoder.fit_transform(X_combined_valid)
    X_recon = encoder.inverse_transform(Z)
    X_recon_surface = X_recon[:, :M*K]

    # compute RMSE
    rmse_surface = np.sqrt(mean_squared_error(X_surface_valid, X_recon_surface))

    # reshape
    X_recon_surface = X_recon_surface.reshape(-1, M, K)
    X_surface_true = X_surface_valid.reshape(-1, M, K)

    print(f"Surface reconstruction RMSE: {rmse_surface:.4f}")

    if return_latent:
        return rmse_surface, X_recon_surface, X_surface_true, Z, valid_mask
    else:
        return rmse_surface, X_recon_surface, X_surface_true
    
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import math


def compare_latent_spaces(
    Z_a,
    Z_b,
    valid_mask=None,
    labels=("Encoder A", "Encoder B"),
    method="tsne",
    return_projection=False,
    full_analysis=False
):
    """
    Compare two latent spaces using t-SNE and/or full latent diagnostics.
    Args:
        Z_a, Z_b (np.ndarray): Latent matrices, shape (T, latent_dim)
        valid_mask (np.ndarray): Optional boolean mask to align valid samples
        labels (tuple): Names for the encoders, e.g., ("PCA", "AE")
        method (str): "tsne" enables global manifold projection
        return_projection (bool): Return 2D t-SNE projection if True
        full_analysis (bool): Plot all latent trajectories and distributions
    Returns:
        np.ndarray (optional): 2D t-SNE projection if requested
    """
    if valid_mask is not None:
        Z_a = Z_a[valid_mask]
        Z_b = Z_b[valid_mask]

    # t-SNE global projection
    if method == "tsne":
        Z_combined = np.vstack([Z_a, Z_b])
        label_array = np.array([labels[0]] * len(Z_a) + [labels[1]] * len(Z_b))
        Z_proj = TSNE(n_components=2, perplexity=30).fit_transform(Z_combined)

        plt.figure(figsize=(6, 5))
        for label in labels:
            idx = label_array == label
            plt.scatter(Z_proj[idx, 0], Z_proj[idx, 1], label=label, alpha=0.6)
        plt.title("t-SNE Projection of Latent Spaces")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # full per-dimension analysis
    if full_analysis:
        latent_dim = Z_a.shape[1]
        n_cols = 3
        n_rows = math.ceil(latent_dim / n_cols)

        # latent trajectories over time
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 2.8 * n_rows), sharex=True)
        axs = axs.flatten()
        for i in range(latent_dim):
            axs[i].plot(Z_a[:, i], label=labels[0], linestyle='--')
            axs[i].plot(Z_b[:, i], label=labels[1], alpha=0.7)
            axs[i].set_title(f"z{i}")
            axs[i].legend()
        plt.suptitle("Latent Trajectories", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

        # latent distributions
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 2.8 * n_rows))
        axs = axs.flatten()
        for i in range(latent_dim):
            sns.kdeplot(Z_a[:, i], ax=axs[i], label=labels[0])
            sns.kdeplot(Z_b[:, i], ax=axs[i], label=labels[1])
            axs[i].set_title(f"z{i} Distribution")
            axs[i].legend()
        plt.suptitle("Latent Distributions", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    if return_projection:
        return Z_proj if method == "tsne" else (Z_a, Z_b)
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd


def summarize_latent_errors(Z_true, Z_pred, name="Model"):
    """
    Summarize latent error metrics: MSE, autocorrelation, variance ratio.
    """
    mse_total = np.mean((Z_true - Z_pred) ** 2)
    mse_per_dim = np.mean((Z_true - Z_pred) ** 2, axis=0)
    acf_lags = 3

    # autocorrs
    acf_true = np.mean([acf(Z_true[:, i], nlags=acf_lags)[1:] for i in range(Z_true.shape[1])], axis=0)
    acf_pred = np.mean([acf(Z_pred[:, i], nlags=acf_lags)[1:] for i in range(Z_pred.shape[1])], axis=0)
    acf_diff = np.mean(np.abs(acf_true - acf_pred))

    # variance retention
    var_ratio = np.var(Z_pred) / np.var(Z_true)

    df = pd.DataFrame([{
        "model": name,
        "latent_mse_total": mse_total,
        "latent_mse_avg_dim": np.mean(mse_per_dim),
        "latent_mse_max_dim": np.max(mse_per_dim),
        "acf_diff_mean": acf_diff,
        "var_ratio_pred/true": var_ratio
    }])

    return df


import pandas as pd


def rank_temporal_model_summaries(summary_df):
    """
    Rank temporal model summaries by error and variance metrics.
    """
    df = summary_df.copy()
    ascending_metrics = ["latent_mse_total", "latent_mse_avg_dim", "latent_mse_max_dim", "acf_diff_mean"]
    descending_metrics = ["var_ratio_pred/true"]

    # rank each column
    for col in ascending_metrics:
        df[f"{col}_rank"] = df[col].rank(method="min", ascending=True)
    for col in descending_metrics:
        df[f"{col}_rank"] = df[col].rank(method="min", ascending=False)

    # mean rank column
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    df["mean_rank"] = df[rank_cols].mean(axis=1)
    df_sorted = df.sort_values("mean_rank").reset_index(drop=True)
    return df_sorted

from keras import ops


    

def evaluate_decoder_output(
    decoder,
    Z_forecast,
    X_grid,
    tensors,
    rel_strikes,
    taus,
    target_date,
    target_tau,
    target_strike,
    X_true_idx=None,
    decoded_surfaces_override=None,
    title="Decoder Evaluation"
):
    """
    Evaluate decoder output against ground truth surfaces.
    Args:
        decoder: decoder model
        Z_forecast: latent vectors
        X_grid: ground truth grid
        tensors: tensor dictionary
        rel_strikes: strike values
        taus: maturity values
        target_date, target_tau, target_strike: for plotting
        X_true_idx: optional index for ground truth
        decoded_surfaces_override: optional override for decoded surfaces
        title: plot title
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import root_mean_squared_error
    from utils.plotting import plot_iv_slices_shifts, plot_smile_slices_comparison
    from keras import ops
    import numpy as np

    # decode or use override
    decoded_surfaces = (
        np.stack([decoder.predict_surface(z) for z in Z_forecast])
        if decoded_surfaces_override is None else np.array(decoded_surfaces_override)
    )

    # ground truth alignment
    if X_true_idx is not None:
        X_true = X_grid[X_true_idx]
    else:
        X_true = X_grid

    # reshape ground truth safely
    try:
        M, K = decoder.M, decoder.K
    except AttributeError:
        M, K = 11, 10  # fallback default

    try:
        X_true = X_true.reshape(-1, M, K)
        decoded_surfaces = decoded_surfaces.reshape(-1, M, K)
    except Exception as e:
        raise ValueError(f"Failed to reshape inputs to ({M}, {K}): {e}")

    # RMSE
    rmse_surface = root_mean_squared_error(
        X_true.reshape(len(X_true), -1),
        decoded_surfaces.reshape(len(decoded_surfaces), -1)
    )
    print(f"{title} RMSE_σ: {rmse_surface:.4f}")

    rmse_per_maturity = np.sqrt(np.mean((X_true - decoded_surfaces)**2, axis=(0,2)))  # shape (M,)
    rmse_per_strike   = np.sqrt(np.mean((X_true - decoded_surfaces)**2, axis=(0,1)))  # shape (K,)
    print(f"RMSE by maturity (mean ± std): {rmse_per_maturity.mean():.4f} ± {rmse_per_maturity.std():.4f}")
    print(f"RMSE by strike    (mean ± std): {rmse_per_strike.mean():.4f} ± {rmse_per_strike.std():.4f}")

    # visualize last surface
    true_surface_t = X_true[-1]
    recon_surface_t = decoded_surfaces[-1]

    plot_iv_slices_shifts(
        true_surface=true_surface_t,
        decoded_surface=recon_surface_t,
        taus=tensors["taus"],
        rel_strikes=rel_strikes,
        decoded_date=target_date,
        target_tau=target_tau,
        target_strike=target_strike,
    )

    plot_smile_slices_comparison(
        true_surface=true_surface_t,
        pred_surface=recon_surface_t,
        rel_strikes=rel_strikes,
        taus=taus,
    )

    # optional penalties
    if hasattr(decoder, "_calendar_penalty"):
        try:
            y_hat = ops.convert_to_tensor(decoded_surfaces.reshape(-1, M * K, 1))
            flat = ops.reshape(y_hat, (-1, 1))
            cal_pen = decoder._calendar_penalty(flat)
            smile_pen = decoder._smile_penalty(flat)
            tau_pen = getattr(decoder, "_tau_penalty", lambda _: ops.zeros(()))(flat)
            print("No-Arbitrage Penalties:")
            print(f"• Calendar: {float(cal_pen):.6f}")
            print(f"• Smile:    {float(smile_pen):.6f}")
            print(f"• Tau:      {float(tau_pen):.6f}")
        except Exception as e:
            print("[Warning] Penalty evaluation skipped:", e)

    # residual RMSE heatmap
    residuals = X_true - decoded_surfaces
    rmse_per_point = np.sqrt(np.mean(np.square(residuals), axis=0))

    plt.figure(figsize=(6, 4))
    plt.imshow(rmse_per_point, cmap="hot", aspect="auto")
    plt.colorbar(label="RMSE")
    plt.title("RMSE Heatmap per (m, τ) Slice")
    plt.xlabel("Strikes")
    plt.ylabel("Maturities")
    plt.xticks(ticks=np.arange(K), labels=np.round(rel_strikes, 2), rotation=90)
    plt.yticks(ticks=np.arange(M), labels=np.round(taus, 2))
    plt.tight_layout()
    plt.show()

def analyze_moe_training_performance(decoder, X_train, X_val, X_test, y_train, y_val, y_test):
   if not decoder.use_moe:
       print("Decoder not using MoE")
       return
   
   print(f"MoE Analysis")
   print(f"Experts: {decoder.num_experts}")
   print(f"Diversity: {decoder.lambda_diversity}")
   
   # analyze gating patterns
   train_analysis = decoder.analyze_gating(X_train)
   val_analysis = decoder.analyze_gating(X_val)
   test_analysis = decoder.analyze_gating(X_test)
   
   # create plots
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   fig.suptitle("MoE Expert Analysis", fontsize=14)
   
   # expert activation frequency
   axes[0, 0].bar(range(decoder.num_experts), train_analysis['expert_activations'], alpha=0.7, label='Train')
   axes[0, 0].bar(range(decoder.num_experts), val_analysis['expert_activations'], alpha=0.7, label='Val')
   axes[0, 0].bar(range(decoder.num_experts), test_analysis['expert_activations'], alpha=0.7, label='Test')
   axes[0, 0].set_title('Expert Activation Frequency')
   axes[0, 0].set_xlabel('Expert ID')
   axes[0, 0].set_ylabel('Average Activation')
   axes[0, 0].legend()
   axes[0, 0].grid(True, alpha=0.3)
   
   # expert specialization with proper biases
   if len(X_train) > 1:
       sample_size = min(10000, len(X_train[0]))
       idx = np.random.choice(len(X_train[0]), sample_size, replace=False)
       
       z_sample = X_train[0][idx]
       m_sample = X_train[1][idx]
       tau_sample = X_train[2][idx]
       
       # compute gate weights with biases
       x_concat = ops.concatenate([z_sample, m_sample, tau_sample], axis=-1)
       for layer in decoder.dense_layers:
           x_concat = layer(x_concat, training=False)
       
       gate_logits = decoder.gating_network(x_concat, training=False)
       if decoder.maturity_specialization:
           expert_biases = decoder._compute_expert_biases(m_sample, tau_sample)
           gate_logits = gate_logits + expert_biases
       gate_weights = ops.convert_to_numpy(ops.nn.softmax(gate_logits, axis=-1))
       
       m_values = ops.convert_to_numpy(m_sample).flatten()
       tau_values = ops.convert_to_numpy(tau_sample).flatten()
       
       # use tab10 colors
       colors = plt.cm.tab10(np.linspace(0, 0.9, decoder.num_experts))
       
       # maturity specialization
       unique_taus = np.sort(np.unique(tau_values))
       for expert_id in range(decoder.num_experts):
           expert_weights = []
           for tau in unique_taus:
               mask = np.abs(tau_values - tau) < 1e-6
               if mask.sum() > 0:
                   expert_weights.append(gate_weights[mask, expert_id].mean())
               else:
                   expert_weights.append(0)
           axes[0, 1].plot(unique_taus, expert_weights, 'o-', 
                      color=colors[expert_id], label=f'E{expert_id}',
                      linewidth=2, markersize=5, alpha=0.8)
       
       axes[0, 1].set_title('Maturity Specialization')
       axes[0, 1].set_xlabel('Maturity (years)')
       axes[0, 1].set_ylabel('Avg Gate Weight')
       axes[0, 1].legend(fontsize=7, ncol=2)
       axes[0, 1].grid(True, alpha=0.3)
       axes[0, 1].set_ylim([0, max(0.5, gate_weights.max()*1.1)])
       
       # strike specialization
       unique_strikes = np.sort(np.unique(m_values))
       for expert_id in range(decoder.num_experts):
           expert_weights = []
           for strike in unique_strikes:
               mask = np.abs(m_values - strike) < 1e-6
               if mask.sum() > 0:
                   expert_weights.append(gate_weights[mask, expert_id].mean())
               else:
                   expert_weights.append(0)
           axes[0, 2].plot(unique_strikes, expert_weights, 'o-',
                         color=colors[expert_id], label=f'E{expert_id}',
                         linewidth=2, markersize=5, alpha=0.8)
       
       axes[0, 2].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
       axes[0, 2].set_title('Strike Specialization')
       axes[0, 2].set_xlabel('Moneyness')
       axes[0, 2].set_ylabel('Avg Gate Weight')
       axes[0, 2].legend(fontsize=7, ncol=2)
       axes[0, 2].grid(True, alpha=0.3)
       axes[0, 2].set_ylim([0, max(0.5, gate_weights.max()*1.1)])
       
       # gate entropy by region
       tau_bins = [0.083, 0.25, 0.5, 1.0, 2.0, 5.0]
       m_bins = [0.6, 0.85, 0.95, 1.05, 1.15, 1.5]
       entropy_matrix = np.zeros((len(tau_bins)-1, len(m_bins)-1))
       
       for i in range(len(tau_bins)-1):
           for j in range(len(m_bins)-1):
               mask = ((tau_values >= tau_bins[i]) & (tau_values < tau_bins[i+1]) &
                      (m_values >= m_bins[j]) & (m_values < m_bins[j+1]))
               if mask.sum() > 0:
                   region_weights = gate_weights[mask]
                   entropy = -np.sum(region_weights * np.log(region_weights + 1e-8), axis=1).mean()
                   entropy_matrix[i, j] = entropy
       
       im = axes[1, 2].imshow(entropy_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=2)
       axes[1, 2].set_title('Gating Entropy by Region')
       axes[1, 2].set_xlabel('Moneyness')
       axes[1, 2].set_ylabel('Maturity')
       axes[1, 2].set_xticks(range(len(m_bins)-1))
       axes[1, 2].set_xticklabels([f'{m:.2f}' for m in (np.array(m_bins[:-1]) + np.array(m_bins[1:]))/2], rotation=45)
       axes[1, 2].set_yticks(range(len(tau_bins)-1))
       axes[1, 2].set_yticklabels(['<0.25y', '0.25-0.5y', '0.5-1y', '1-2y', '>2y'])
       plt.colorbar(im, ax=axes[1, 2], label='Entropy')
       
       for i in range(len(tau_bins)-1):
           for j in range(len(m_bins)-1):
               axes[1, 2].text(j, i, f'{entropy_matrix[i, j]:.2f}', 
                           ha='center', va='center',
                           color='white' if entropy_matrix[i, j] > 1.0 else 'black',
                           fontsize=8)
   
   # expert correlation matrix
   n_samples = min(1000, len(X_train[0]))
   x_processed = ops.concatenate([X_train[0][:n_samples], 
                                 X_train[1][:n_samples], 
                                 X_train[2][:n_samples]], axis=-1)
   for layer in decoder.dense_layers:
       x_processed = layer(x_processed, training=False)
   
   expert_outputs_raw = []
   for expert in decoder.experts:
       output = expert(x_processed, training=False)
       expert_outputs_raw.append(ops.convert_to_numpy(output).flatten())
   
   correlation_matrix = np.corrcoef(expert_outputs_raw)
   im = axes[1, 0].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
   axes[1, 0].set_title('Expert Output Correlation')
   axes[1, 0].set_xlabel('Expert ID')
   axes[1, 0].set_ylabel('Expert ID')
   
   for i in range(decoder.num_experts):
       for j in range(decoder.num_experts):
           axes[1, 0].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                          ha='center', va='center', 
                          color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black',
                          fontsize=8)
   plt.colorbar(im, ax=axes[1, 0])
   
   # expert prediction distributions
   sample_predictions = []
   for expert in decoder.experts:
       pred = expert(x_processed, training=False)
       sample_predictions.append(ops.convert_to_numpy(pred).flatten())
   
   for i, pred in enumerate(sample_predictions):
       axes[1, 1].hist(pred, bins=20, alpha=0.6, label=f'Expert {i}', 
                      density=True, color=colors[i])
   
   axes[1, 1].set_title('Expert Prediction Distributions')
   axes[1, 1].set_xlabel('Predicted IV')
   axes[1, 1].set_ylabel('Density')
   axes[1, 1].legend(fontsize=8)
   axes[1, 1].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()
   
   print(f"\nSummary:")
   print(f"Most active expert: {train_analysis['expert_activations'].argmax()}")
   print(f"Least active expert: {train_analysis['expert_activations'].argmin()}")
   print(f"Balance score: {1 - np.std(train_analysis['expert_activations']):.3f}")
   print(f"Avg entropy: {train_analysis['gate_entropy'].mean():.3f}")
   if correlation_matrix.size > 1:
       print(f"Min correlation: {correlation_matrix[correlation_matrix < 1].min():.3f}")
       print(f"Max correlation: {correlation_matrix[correlation_matrix < 1].max():.3f}")
   
   return {
       'train_analysis': train_analysis,
       'val_analysis': val_analysis, 
       'test_analysis': test_analysis,
       'expert_correlations': correlation_matrix
   }
   
   
def evaluate_decoder_output_v2(
    Y_pred,                    # Pre-computed predictions (N, M, K)
    Y_true,                    # Ground truth surfaces (N, M*K) or (N, M, K)
    tensors,                   # Tensor dictionary
    rel_strikes,               # Strike values
    taus,                      # Maturity values
    target_date,               # Target date for plotting
    target_tau,                # Target tau for plotting
    target_strike,             # Target strike for plotting
    decoder=None,              # Optional: for penalty calculation
    title="Decoder Evaluation"
):
    """
    Evaluate decoder predictions against ground truth.
    Args:
        Y_pred: array (N, M, K) - decoder predictions
        Y_true: array (N, M*K) or (N, M, K) - ground truth surfaces
        tensors: dict - tensor dictionary containing 'taus'
        rel_strikes: array - strike values
        taus: array - maturity values
        target_date: str - date for plotting
        target_tau: float - tau for plotting
        target_strike: float - strike for plotting
        decoder: optional - decoder model for penalty calculation
        title: str - plot title
    Returns:
        dict with RMSE metrics
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import root_mean_squared_error
    from keras import ops
    import numpy as np

    # ensure predictions are numpy arrays
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    # get dimensions from predictions
    if len(Y_pred.shape) != 3:
        raise ValueError(f"Y_pred must be 3D (N, M, K), got shape {Y_pred.shape}")

    N, M, K = Y_pred.shape
    print(f"Evaluation dimensions: N={N}, M={M}, K={K}")

    # reshape ground truth to match predictions
    if len(Y_true.shape) == 2:
        if Y_true.shape[1] != M * K:
            raise ValueError(f"Y_true shape {Y_true.shape} incompatible with Y_pred {Y_pred.shape}")
        Y_true = Y_true.reshape(N, M, K)
    elif len(Y_true.shape) == 3:
        if Y_true.shape != Y_pred.shape:
            raise ValueError(f"Y_true shape {Y_true.shape} != Y_pred shape {Y_pred.shape}")
    else:
        raise ValueError(f"Y_true must be 2D (N, M*K) or 3D (N, M, K), got shape {Y_true.shape}")

    # RMSE metrics
    rmse_surface = root_mean_squared_error(
        Y_true.reshape(N, -1),
        Y_pred.reshape(N, -1)
    )
    print(f"{title} RMSE_σ: {rmse_surface:.4f}")

    rmse_per_maturity = np.sqrt(np.mean((Y_true - Y_pred)**2, axis=(0,2)))  # shape (M,)
    rmse_per_strike   = np.sqrt(np.mean((Y_true - Y_pred)**2, axis=(0,1)))  # shape (K,)
    print(f"RMSE by maturity (mean ± std): {rmse_per_maturity.mean():.4f} ± {rmse_per_maturity.std():.4f}")
    print(f"RMSE by strike    (mean ± std): {rmse_per_strike.mean():.4f} ± {rmse_per_strike.std():.4f}")

    # visualize last surface
    true_surface_t = Y_true[-1]      # (M, K)
    pred_surface_t = Y_pred[-1]      # (M, K)

    print(f"Plotting last surface shapes: true={true_surface_t.shape}, pred={pred_surface_t.shape}")

    plot_iv_slices_shifts_v2(
        true_surface=true_surface_t,
        decoded_surface=pred_surface_t,
        taus=tensors["taus"],
        rel_strikes=rel_strikes,
        decoded_date=target_date,
        target_tau=target_tau,
        target_strike=target_strike,
    )

    plot_smile_slices_comparison_v2(
        true_surface=true_surface_t,
        pred_surface=pred_surface_t,
        rel_strikes=rel_strikes,
        taus=taus,
    )

    # optional penalties (if decoder provided)
    if decoder is not None and hasattr(decoder, "_calendar_penalty"):
        try:
            y_hat = ops.convert_to_tensor(Y_pred.reshape(-1, M * K, 1))
            flat = ops.reshape(y_hat, (-1, 1))
            cal_pen = decoder._calendar_penalty(flat)
            smile_pen = decoder._smile_penalty(flat)
            tau_pen = getattr(decoder, "_tau_penalty", lambda _: ops.zeros(()))(flat)
            print("No-Arbitrage Penalties:")
            print(f"• Calendar: {float(cal_pen):.6f}")
            print(f"• Smile:    {float(smile_pen):.6f}")
            print(f"• Tau:      {float(tau_pen):.6f}")
        except Exception as e:
            print("[Warning] Penalty evaluation skipped:", e)

    # residual RMSE heatmap
    residuals = Y_true - Y_pred
    rmse_per_point = np.sqrt(np.mean(np.square(residuals), axis=0))  # (M, K)

    plt.figure(figsize=(6, 4))
    plt.imshow(rmse_per_point, cmap="hot", aspect="auto", origin="lower")
    plt.colorbar(label="RMSE")
    plt.title("RMSE Heatmap per (m, τ) Slice")
    plt.xlabel("Strikes")
    plt.ylabel("Maturities")
    plt.xticks(ticks=np.arange(K), labels=np.round(rel_strikes, 2), rotation=90)
    plt.yticks(ticks=np.arange(M), labels=np.round(taus, 2))
    plt.tight_layout()
    plt.show()

    return {
        'rmse_overall': rmse_surface,
        'rmse_by_maturity': rmse_per_maturity,
        'rmse_by_strike': rmse_per_strike
    }
    import matplotlib.pyplot as plt

# --- MoE Training Analysis and Evaluation ---




import numpy as np
import matplotlib.pyplot as plt

def plot_iv_slices_shifts_v2(
    true_surface,
    decoded_surface,
    rel_strikes,
    taus,
    decoded_date,
    target_tau,
    target_strike,
    strike_shifts=None,
    tau_shifts=None,
    label_decoded="Decoded"
):
    if strike_shifts is None:
        strike_shifts = [-0.3, 0.0, 0.3]
    if tau_shifts is None:
        tau_shifts = [-0.5, 0.0, 0.5]

    if len(strike_shifts) != len(tau_shifts):
        raise ValueError("strike_shifts and tau_shifts must be the same length")

    taus = np.asarray(taus)
    rel_strikes = np.asarray(rel_strikes)
    n = len(strike_shifts)
    fig, axs = plt.subplots(n, 2, figsize=(14, 4 * n))

    for i in range(n):
        k_shift = strike_shifts[i]
        tau_shift = tau_shifts[i]

        # --- Smile slice
        tau_val = target_tau + tau_shift
        tau_idx = np.argmin(np.abs(taus - tau_val))
        smile_true = true_surface[tau_idx, :]
        smile_pred = decoded_surface[tau_idx, :]
        valid_smile = ~np.isnan(smile_true) & ~np.isnan(smile_pred)

        axs[i, 0].plot(rel_strikes[valid_smile], smile_true[valid_smile], label="true", marker='o')
        axs[i, 0].plot(rel_strikes[valid_smile], smile_pred[valid_smile], label=label_decoded, marker='.')
        axs[i, 0].set_title(f"Smile @ τ ≈ {taus[tau_idx]:.2f} (shift {tau_shift:+.2f})")
        axs[i, 0].set_xlabel("rel_strike")
        axs[i, 0].set_ylabel("IV")
        axs[i, 0].legend()

        # --- Term structure slice
        strike_val = target_strike + k_shift
        k_idx = np.argmin(np.abs(rel_strikes - strike_val))
        term_true = true_surface[:, k_idx]
        term_pred = decoded_surface[:, k_idx]
        valid_term = ~np.isnan(term_true) & ~np.isnan(term_pred)

        axs[i, 1].plot(taus[valid_term], term_true[valid_term], label="true", marker='o')
        axs[i, 1].plot(taus[valid_term], term_pred[valid_term], label=label_decoded, marker='.')
        axs[i, 1].set_title(f"Term @ strike ≈ {rel_strikes[k_idx]:.2f} (shift {k_shift:+.2f})")
        axs[i, 1].set_xlabel("Tau")
        axs[i, 1].set_ylabel("IV")
        axs[i, 1].legend()

    plt.suptitle(f"{label_decoded} Surface vs True — {decoded_date}", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_smile_slices_comparison_v2(true_surface, pred_surface, rel_strikes, taus, title="True vs Predicted Smile Slices"):
    true_surface = np.asarray(true_surface)
    pred_surface = np.asarray(pred_surface)
    rel_strikes = np.asarray(rel_strikes)
    taus = np.asarray(taus)
    
    M, K = true_surface.shape
    assert pred_surface.shape == (M, K), f"Shape mismatch: true {true_surface.shape} vs pred {pred_surface.shape}"
    assert len(rel_strikes) == K, f"rel_strikes length {len(rel_strikes)} != K {K}"
    assert len(taus) == M, f"taus length {len(taus)} != M {M}"
    
    ncols = 4
    nrows = int(np.ceil(M / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3 * nrows))
    if nrows == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()

    for i in range(M):
        smile_true = true_surface[i, :]
        smile_pred = pred_surface[i, :]
        valid_mask = ~np.isnan(smile_true) & ~np.isnan(smile_pred)
        
        axs[i].plot(rel_strikes[valid_mask], smile_true[valid_mask], label="True", marker='o')
        axs[i].plot(rel_strikes[valid_mask], smile_pred[valid_mask], label="Pred", marker='x')
        axs[i].set_title(f"Smile @ τ={taus[i]:.2f}")
        axs[i].set_xlabel("log-moneyness")
        axs[i].set_ylabel("IV")
        axs[i].grid(True)
        axs[i].legend()

    for j in range(M, len(axs)):
        axs[j].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()