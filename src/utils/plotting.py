import matplotlib.pyplot as plt
import numpy as np

# plot functions for visualizing implied volatility surfaces and slices taken from various notebook 
# # to rearrange!!

def plot_surface_snapshot(iv_matrix, taus, rel_strikes, decoded_date, target_tau=1.0, target_strike=1.0):
    taus = np.array(taus)
    rel_strikes = np.array(rel_strikes)

    m_idx = np.argmin(np.abs(taus - target_tau))
    try:
        k_idx = np.where(np.round(rel_strikes, 3) == round(target_strike, 3))[0][0]
    except IndexError:
        k_idx = np.argmin(np.abs(rel_strikes - target_strike))
        print("rel_strike not found exactly — using closest.")

    smile = iv_matrix[m_idx]
    term = iv_matrix[:, k_idx]
    valid_smile = ~np.isnan(smile)
    valid_term = ~np.isnan(term)
    #IV_YLIM = (0.0, 0.6)  # dataset range allows


    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(rel_strikes[valid_smile], smile[valid_smile], marker='o')
    axs[0].set_title(f"Smile @ τ = {taus[m_idx]:.2f}", fontsize=10)
    axs[0].set_xlabel("rel_strike", fontsize=10)
    axs[0].set_ylabel("IV", fontsize=10)
    #axs[0].set_ylim(*IV_YLIM)

    axs[1].plot(taus[valid_term], term[valid_term], marker='o')
    axs[1].set_title(f"Term Structure @ rel_strike = {rel_strikes[k_idx]:.2f}", fontsize=10)
    axs[1].set_xlabel("Tau (years)", fontsize=10)
    axs[1].set_ylabel("IV", fontsize=10)
    #axs[1].set_ylim(*IV_YLIM)

    plt.suptitle(f"Surface Snapshot — {decoded_date}", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_raw_iv_slice(df, target_date, target_strike=1.0, target_tau=1.0):
    import pandas as pd

    df_day = df[df["date"] == pd.to_datetime(target_date)].copy()
    if df_day.empty:
        raise ValueError("No data found for target date")

    rel_strikes = sorted(df_day["rel_strike"].dropna().unique())
    taus = sorted(df_day["tau"].dropna().unique())
    taus_arr = np.array(taus)
    rel_strikes_arr = np.array(rel_strikes)

    m_idx = np.argmin(np.abs(taus_arr - target_tau))
    df_smile = df_day[np.isclose(df_day["tau"], taus_arr[m_idx])]
    smile_vals = df_smile[["rel_strike", "market_iv"]].dropna().values

    k_idx = np.argmin(np.abs(rel_strikes_arr - target_strike))
    df_term = df_day[np.isclose(df_day["rel_strike"], rel_strikes_arr[k_idx])]
    term_vals = df_term[["tau", "market_iv"]].dropna().values
    #IV_YLIM = (0.0, 0.6)  # dataset range allows

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    if len(smile_vals) > 0:
        axs[0].plot(smile_vals[:, 0], smile_vals[:, 1], marker='o')
    axs[0].set_title(f"Smile @ τ ≈ {taus_arr[m_idx]:.2f}")
    axs[0].set_xlabel("rel_strike")
    axs[0].set_ylabel("IV")
    #axs[0].set_ylim(*IV_YLIM)

    if len(term_vals) > 0:
        axs[1].plot(term_vals[:, 0], term_vals[:, 1], marker='o')
    axs[1].set_title(f"Term Structure @ rel_strike ≈ {rel_strikes_arr[k_idx]:.2f}")
    axs[1].set_xlabel("Tau (years)")
    axs[1].set_ylabel("IV")
    #axs[1].set_ylim(*IV_YLIM)

    plt.suptitle(f"Raw IV Surface — {target_date}", fontsize=13)
    plt.tight_layout()
    plt.show()
    


def plot_decoded_surface_vector(X_true, X_decoded, index=-1, title_prefix="Decoded vs True"):
    index = min(len(X_true), len(X_decoded)) + index if index < 0 else index
    #IV_YLIM = (0.0, 0.6)  # dataset range allows
    plt.figure(figsize=(12, 4))
    plt.plot(X_true[index], label="true")
    plt.plot(X_decoded[index], label="decoded")
    #plt.ylim(*IV_YLIM)
    plt.title(f"{title_prefix} (t={index})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_iv_slices(true_surface, decoded_surface, taus, rel_strikes, decoded_date,
                   target_tau=1.0, target_strike=1.0, label_decoded="decoded"):
    taus = np.array(taus)
    rel_strikes = np.array(rel_strikes)

    m_idx = np.argmin(np.abs(taus - target_tau))
    try:
        k_idx = np.where(np.round(rel_strikes, 3) == round(target_strike, 3))[0][0]
    except IndexError:
        k_idx = np.argmin(np.abs(rel_strikes - target_strike))
        print("rel_strike not found exactly — using closest.")

    smile_true = true_surface[m_idx]
    smile_decoded = decoded_surface[m_idx]
    term_true = true_surface[:, k_idx]
    term_decoded = decoded_surface[:, k_idx]

    valid_smile = ~np.isnan(smile_true) & ~np.isnan(smile_decoded)
    valid_term = ~np.isnan(term_true) & ~np.isnan(term_decoded)
    IV_YLIM = (0.0, 0.6)  # dataset range allows

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(rel_strikes[valid_smile], smile_true[valid_smile], label="true", marker='o')
    axs[0].plot(rel_strikes[valid_smile], smile_decoded[valid_smile], label=label_decoded, marker='.')
    axs[0].set_title(f"Smile @ τ = {taus[m_idx]:.2f}")
    axs[0].set_xlabel("rel_strike")
    axs[0].set_ylabel("IV")
    #axs[0].set_ylim(*IV_YLIM)
    axs[0].legend()

    axs[1].plot(taus[valid_term], term_true[valid_term], label="true", marker='o')
    axs[1].plot(taus[valid_term], term_decoded[valid_term], label=label_decoded, marker='.')
    axs[1].set_title(f"Term Structure @ rel_strike = {rel_strikes[k_idx]:.2f}")
    axs[1].set_xlabel("Tau")
    axs[1].set_ylabel("IV")
    #axs[1].set_ylim(*IV_YLIM)
    axs[1].legend()

    plt.suptitle(f"{label_decoded.capitalize()} Surface vs True — {decoded_date}", fontsize=13)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_iv_slices_shifts(
    true_surface,
    decoded_surface,
    taus,
    rel_strikes,
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
        smile_true = true_surface[tau_idx]
        smile_pred = decoded_surface[tau_idx]
        valid_smile = ~np.isnan(smile_true)
       # IV_YLIM = (0.0, 0.6)  # dataset range allows

        axs[i, 0].plot(rel_strikes[valid_smile], smile_true[valid_smile], label="true", marker='o')
        axs[i, 0].plot(rel_strikes[valid_smile], smile_pred[valid_smile], label=label_decoded, marker='.')
        axs[i, 0].set_title(f"Smile @ τ ≈ {taus[tau_idx]:.2f} (shift {tau_shift:+.2f})")
        axs[i, 0].set_xlabel("rel_strike")
        axs[i, 0].set_ylabel("IV")
        #axs[i, 0].set_ylim(*IV_YLIM)
        axs[i, 0].legend()

        # --- Term structure slice
        strike_val = target_strike + k_shift
        k_idx = np.argmin(np.abs(rel_strikes - strike_val))
        term_true = true_surface[:, k_idx]
        term_pred = decoded_surface[:, k_idx]
        valid_term = ~np.isnan(term_true)

        axs[i, 1].plot(taus[valid_term], term_true[valid_term], label="true", marker='o')
        axs[i, 1].plot(taus[valid_term], term_pred[valid_term], label=label_decoded, marker='.')
        axs[i, 1].set_title(f"Term @ strike ≈ {rel_strikes[k_idx]:.2f} (shift {k_shift:+.2f})")
        axs[i, 1].set_xlabel("Tau")
        axs[i, 1].set_ylabel("IV")
        #axs[i, 1].set_ylim(*IV_YLIM)
        axs[i, 1].legend()

    plt.suptitle(f"{label_decoded} Surface vs True — {decoded_date}", fontsize=16)
    plt.tight_layout()
    plt.show()
   
    


import matplotlib.pyplot as plt
import numpy as np

def plot_smile_slices_comparison(true_surface, pred_surface, rel_strikes, taus, title="True vs Predicted Smile Slices"):
    """
    Plots smile slices for all maturities comparing true and predicted surfaces.

    Args:
        true_surface (np.ndarray): shape (M, K)
        pred_surface (np.ndarray): shape (M, K)
        rel_strikes (np.ndarray): strike grid (K,)
        taus (np.ndarray): maturity grid (M,)
        title (str): optional plot title
    """
    M, K = true_surface.shape
    ncols = 4
    nrows = int(np.ceil(M / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3 * nrows))
    axs = axs.flatten()
    #IV_YLIM = (0.0, 0.6)  # dataset range allows
    for i in range(M):
        axs[i].plot(rel_strikes, true_surface[i], label="True", marker='o')
        axs[i].plot(rel_strikes, pred_surface[i], label="Pred", marker='x')
        axs[i].set_title(f"Smile @ τ={taus[i]:.2f}")
        axs[i].set_xlabel("log-moneyness")
        axs[i].set_ylabel("IV")
        #axs[i].set_ylim(*IV_YLIM)
        axs[i].grid(True)
        axs[i].legend()

    for j in range(M, len(axs)):
        axs[j].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from sklearn.manifold import TSNE
import seaborn as sns

def plot_latent_diagnostics(Z_true, Z_pred, name=None, tsne=False):
    latent_dim = Z_true.shape[1]
    name = name or "Latent Space"
    
    # --- 1. Per-dimension MSE
    mse_per_dim = np.mean((Z_true - Z_pred) ** 2, axis=0)
    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(latent_dim), mse_per_dim)
    plt.title(f"{name}: MSE per Latent Dimension")
    plt.xlabel("Latent Dimension")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()

    # --- 2. Temporal traces
    n_cols = 3
    n_rows = int(np.ceil(latent_dim / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axs = axs.flatten()
    for i in range(latent_dim):
        axs[i].plot(Z_true[:, i], label="True", alpha=0.7)
        axs[i].plot(Z_pred[:, i], "--", label="Pred", alpha=0.7)
        axs[i].set_title(f"z[{i}]")
        axs[i].legend()
    plt.suptitle(f"{name}: Temporal Comparison z(t+1)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- 3. Autocorrelations
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axs = axs.flatten()
    for i in range(latent_dim):
        acf_true = acf(Z_true[:, i], nlags=10)
        acf_pred = acf(Z_pred[:, i], nlags=10)
        axs[i].plot(acf_true, label="True")
        axs[i].plot(acf_pred, label="Pred", linestyle="--")
        axs[i].set_title(f"ACF z[{i}]")
        axs[i].legend()
    plt.suptitle(f"{name}: Autocorrelations", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- 4. Error heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(np.abs(Z_true - Z_pred), cmap="BuPu", cbar=True, vmin=0, vmax=0.2 )
    plt.title(f"{name}: Absolute Error Heatmap")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.show()

    # --- 5. Optional t-SNE comparison
    if tsne:
        Z_all = np.concatenate([Z_true, Z_pred])
        Z_proj = TSNE(n_components=2, perplexity=30).fit_transform(Z_all)
        N = len(Z_true)
        plt.figure(figsize=(6, 5))
        plt.scatter(Z_proj[:N, 0], Z_proj[:N, 1], label="True", alpha=0.5)
        plt.scatter(Z_proj[N:, 0], Z_proj[N:, 1], label="Pred", alpha=0.5)
        plt.legend()
        plt.title(f"{name}: t-SNE of Latents")
        plt.tight_layout()
        plt.show()
def plot_ssvi_iv_slice(df, target_date, target_strike=1.0, target_tau=1.0):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df_day = df[df["date"] == pd.to_datetime(target_date)].copy()
    if df_day.empty:
        raise ValueError("No data found for target date")

    rel_strikes = sorted(df_day["rel_strike"].dropna().unique())
    taus = sorted(df_day["tau"].dropna().unique())
    taus_arr = np.array(taus)
    rel_strikes_arr = np.array(rel_strikes)

    # Get smile slice (fixed tau, varying strikes)
    m_idx = np.argmin(np.abs(taus_arr - target_tau))
    df_smile = df_day[np.isclose(df_day["tau"], taus_arr[m_idx])]
    smile_vals = df_smile[["rel_strike", "ssvi_reconstructed"]].dropna().values

    # Get term structure slice (fixed strike, varying tau)
    k_idx = np.argmin(np.abs(rel_strikes_arr - target_strike))
    df_term = df_day[np.isclose(df_day["rel_strike"], rel_strikes_arr[k_idx])]
    term_vals = df_term[["tau", "ssvi_reconstructed"]].dropna().values

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot smile
    if len(smile_vals) > 0:
        axs[0].plot(smile_vals[:, 0], smile_vals[:, 1], marker='o', color='orange')
    axs[0].set_title(f"SSVI Smile @ τ ≈ {taus_arr[m_idx]:.2f}")
    axs[0].set_xlabel("rel_strike")
    axs[0].set_ylabel("IV")
    axs[0].grid(True)

    # Plot term structure
    if len(term_vals) > 0:
        axs[1].plot(term_vals[:, 0], term_vals[:, 1], marker='o', color='orange')
    axs[1].set_title(f"SSVI Term Structure @ rel_strike ≈ {rel_strikes_arr[k_idx]:.2f}")
    axs[1].set_xlabel("Tau (years)")
    axs[1].set_ylabel("IV")
    axs[1].grid(True)

    plt.suptitle(f"SSVI Reconstructed IV Surface — {target_date}", fontsize=13)
    plt.tight_layout()
    plt.show()
    
    
    