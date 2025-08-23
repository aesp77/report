# Evaluation utilities
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
import pandas as pd
from keras import ops

def evaluate_forecast(X_true, X_decoded):
    """Basic forecast evaluation metrics."""
    n = min(X_true.shape[0], X_decoded.shape[0])
    X_true, X_decoded = X_true[:n], X_decoded[:n]
    mse = mean_squared_error(X_true, X_decoded)
    mae = mean_absolute_error(X_true, X_decoded)
    return mse, mae, X_true, X_decoded

def print_evaluation(mse, mae, label="Forecast"):
    print(f"{label} MSE: {mse:.6f} | MAE: {mae:.6f}")

def evaluate_pca_var(pca, Z_train, Z_test, Z_pred, n_components=None,
                      plot_time_series=True, plot_residuals=True):
    """PCA diagnostics and component-wise forecast evaluation."""
    if n_components is None:
        n_components = Z_pred.shape[1]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Scree plot
    axs[0, 0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    axs[0, 0].set_xlabel("Number of PCA components")
    axs[0, 0].set_ylabel("Cumulative explained variance")
    axs[0, 0].set_title("PCA Scree Plot")
    axs[0, 0].grid(True)

    # Latent space: training vs test
    axs[0, 1].scatter(Z_train[:, 0], Z_train[:, 1], label="Train", alpha=0.6)
    axs[0, 1].scatter(Z_test[:, 0], Z_test[:, 1], label="Test", alpha=0.6)
    axs[0, 1].set_title("PCA Latent Space")
    axs[0, 1].set_xlabel("PC1")
    axs[0, 1].set_ylabel("PC2")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Forecast vs True in PC space
    axs[1, 0].scatter(Z_test[:len(Z_pred), 0], Z_test[:len(Z_pred), 1], label="True", alpha=0.5)
    axs[1, 0].scatter(Z_pred[:, 0], Z_pred[:, 1], label="Forecast", alpha=0.5)
    axs[1, 0].set_title("Forecast vs True Trajectories (PC1-PC2)")
    axs[1, 0].set_xlabel("PC1")
    axs[1, 0].set_ylabel("PC2")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # MSE per component
    mse_vals = [mean_squared_error(Z_test[:len(Z_pred), i], Z_pred[:, i]) for i in range(n_components)]
    axs[1, 1].bar([f"PC{i+1}" for i in range(n_components)], mse_vals)
    axs[1, 1].set_title("MSE per PCA Component")
    axs[1, 1].set_ylabel("MSE")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Time series plots per component
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

    # Residual heatmap
    if plot_residuals:
        residuals = Z_test[:len(Z_pred)] - Z_pred
        plt.figure(figsize=(12, 4))
        sns.heatmap(residuals.T, cmap="coolwarm", center=0, cbar_kws={"label": "Residual"})
        plt.xlabel("Forecast Step")
        plt.ylabel("PCA Component")
        plt.title("Residual Heatmap: Z_true - Z_pred")
        plt.tight_layout()
        plt.show()

def evaluate_encoder(encoder, X_surface, X_features, M, K, return_latent=False):
    """Evaluate encoder reconstruction quality."""
    X_combined = np.hstack([X_surface, X_features])

    # Filter valid rows
    valid_mask = np.all(np.isfinite(X_combined), axis=1)
    X_combined_valid = X_combined[valid_mask]
    X_surface_valid = X_surface[valid_mask]

    # Fit and decode
    Z = encoder.fit_transform(X_combined_valid)
    X_recon = encoder.inverse_transform(Z)
    X_recon_surface = X_recon[:, :M*K]

    # Compute RMSE
    rmse_surface = np.sqrt(mean_squared_error(X_surface_valid, X_recon_surface))

    # Reshape
    X_recon_surface = X_recon_surface.reshape(-1, M, K)
    X_surface_true = X_surface_valid.reshape(-1, M, K)

    print(f"Surface reconstruction RMSE: {rmse_surface:.4f}")

    if return_latent:
        return rmse_surface, X_recon_surface, X_surface_true, Z, valid_mask
    else:
        return rmse_surface, X_recon_surface, X_surface_true

def summarize_latent_errors(Z_true, Z_pred, name="Model"):
    """Compute latent space error metrics."""
    mse_total = np.mean((Z_true - Z_pred) ** 2)
    mse_per_dim = np.mean((Z_true - Z_pred) ** 2, axis=0)
    acf_lags = 3

    # Autocorrelations
    acf_true = np.mean([acf(Z_true[:, i], nlags=acf_lags)[1:] for i in range(Z_true.shape[1])], axis=0)
    acf_pred = np.mean([acf(Z_pred[:, i], nlags=acf_lags)[1:] for i in range(Z_pred.shape[1])], axis=0)
    acf_diff = np.mean(np.abs(acf_true - acf_pred))

    # Variance retention
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

def rank_temporal_model_summaries(summary_df):
    """Rank temporal models by multiple metrics."""
    df = summary_df.copy()
    
    # Define metric directions
    ascending_metrics = ["latent_mse_total", "latent_mse_avg_dim", "latent_mse_max_dim", "acf_diff_mean"]
    descending_metrics = ["var_ratio_pred/true"]

    # Rank each column
    for col in ascending_metrics:
        df[f"{col}_rank"] = df[col].rank(method="min", ascending=True)

    for col in descending_metrics:
        df[f"{col}_rank"] = df[col].rank(method="min", ascending=False)

    # Add mean rank
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    df["mean_rank"] = df[rank_cols].mean(axis=1)

    # Sort by overall rank
    df_sorted = df.sort_values("mean_rank").reset_index(drop=True)
    return df_sorted

def plot_iv_slices_shifts_v2(true_surface, decoded_surface, rel_strikes, taus,
                             decoded_date, target_tau, target_strike,
                             strike_shifts=None, tau_shifts=None, label_decoded="Decoded"):
    """Plot smile and term structure slices."""
    if strike_shifts is None:
        strike_shifts = [-0.3, 0.0, 0.3]
    if tau_shifts is None:
        tau_shifts = [-0.5, 0.0, 0.5]

    taus = np.asarray(taus)
    rel_strikes = np.asarray(rel_strikes)
    n = len(strike_shifts)
    fig, axs = plt.subplots(n, 2, figsize=(14, 4 * n))

    for i in range(n):
        k_shift = strike_shifts[i]
        tau_shift = tau_shifts[i]

        # Smile slice
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

        # Term structure slice
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

def plot_smile_slices_comparison_v2(true_surface, pred_surface, rel_strikes, taus, 
                                   title="True vs Predicted Smile Slices"):
    """Compare smile slices across maturities."""
    true_surface = np.asarray(true_surface)
    pred_surface = np.asarray(pred_surface)
    rel_strikes = np.asarray(rel_strikes)
    taus = np.asarray(taus)
    
    M, K = true_surface.shape
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