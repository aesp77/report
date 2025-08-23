import numpy as np
from keras import ops
from scipy.interpolate import interp1d

def add_tau_features_tensor(surface_tensor, taus):
    """Adds engineered tau features to surface tensor."""
    T, M, K, _ = surface_tensor.shape
    tau_raw = np.array(taus).reshape(1, M, 1)
    tau_raw = np.tile(tau_raw, (T, 1, K))

    sqrt_tau = np.sqrt(tau_raw + 1e-4)
    log_tau = np.log(tau_raw + 1e-4)
    inv_tau = 1 / (tau_raw + 1e-4)
    norm_tau = (tau_raw - np.mean(tau_raw)) / (np.std(tau_raw) + 1e-4)

    tau_feat = np.stack([tau_raw, sqrt_tau, log_tau, inv_tau, norm_tau], axis=-1)
    return ops.convert_to_tensor(tau_feat.astype("float32"))

def add_iv_moving_averages(surface_tensor, windows=[5, 10, 20]):
    """Adds IV moving average features to surface tensor."""
    surface_np = ops.convert_to_numpy(surface_tensor)
    T, M, K, C = surface_np.shape

    iv = surface_np[..., -1]
    iv_ma_features = []

    for w in windows:
        ma = np.zeros_like(iv)
        for t in range(T):
            if t < w:
                ma[t] = iv[t]
            else:
                ma[t] = np.mean(iv[t-w:t], axis=0)
        iv_ma_features.append(ma[..., np.newaxis])

    ma_tensor = np.concatenate(iv_ma_features, axis=-1)
    combined = np.concatenate([surface_np, ma_tensor], axis=-1)
    return ops.convert_to_tensor(combined.astype("float32"))

def interpolate_monthly_maturities(surface_tensor, taus, monthly_maturities):
    """Interpolates surface tensor to include monthly maturities."""
    surface_np = ops.convert_to_numpy(surface_tensor)
    T, M, K, C = surface_np.shape

    existing_maturities = set(taus)
    missing_maturities = [m for m in monthly_maturities if m not in existing_maturities]
    all_maturities = np.array(sorted(list(existing_maturities) + missing_maturities))

    interpolated_surface = np.zeros((T, len(all_maturities), K, C))
    for t in range(T):
        for k in range(K):
            for c in range(C):
                interp_func = interp1d(taus, surface_np[t, :, k, c], kind='linear', fill_value='extrapolate')
                interpolated_surface[t, :, k, c] = interp_func(all_maturities)

    interpolated_tensor = ops.convert_to_tensor(interpolated_surface.astype("float32"))
    return interpolated_tensor, all_maturities



#################### BAELINE FEATURES ####################

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def compute_log_returns(df, column="S0"):
    """Computes log returns for specified column."""
    returns = np.log(df[column]).diff()
    return returns.rename("log_return")

def compute_realized_vol(returns, window=5):
    """Computes realized volatility over rolling window."""
    realized = returns.rolling(window).std()
    return realized.rename(f"realized_vol_{window}d")


def compute_rolling_features(series, window=20):
    """Computes rolling mean and z-score features."""
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    zscore = (series - roll_mean) / roll_std
    return roll_mean.rename(f"{series.name}_ma{window}"), zscore.rename(f"{series.name}_z{window}")


def compress_yield_curve(df, rf_cols):
    """Compresses yield curve into key rate and slope features."""
    df = df.copy()
    rf_numeric = df[rf_cols].ffill().bfill()

    tenors = [float(col.replace("rf_", "")) for col in rf_numeric.columns]
    rf_numeric.columns = tenors
    rf_sorted = rf_numeric.sort_index(axis=1)

    rates = {}
    for t in [0.25, 0.5, 1, 2, 3, 4, 5]:
        if t in rf_sorted.columns:
            rates[f"rate_{t}y"] = rf_sorted[t]
        else:
            closest = min(rf_sorted.columns, key=lambda x: abs(x - t))
            rates[f"rate_{t}y"] = rf_sorted[closest]

    rate_10y = rf_sorted.loc[:, (rf_sorted.columns > 5.0) & (rf_sorted.columns <= 10.0)].mean(axis=1)
    rate_30y = rf_sorted.loc[:, rf_sorted.columns > 10.0].mean(axis=1)

    slope_5_10 = rate_10y - rates["rate_5y"]
    slope_10_30 = rate_30y - rate_10y

    return pd.DataFrame({
        **rates,
        "rate_10y": rate_10y,
        "rate_30y": rate_30y,
        "slope_5_10": slope_5_10,
        "slope_10_30": slope_10_30,
    }, index=df.index)


def compute_smile_features_by_maturity(df, target_strikes=[0.8, 1.0, 1.2]):
    """Computes smile skew and convexity per (date, maturity)."""
    rows = []
    grouped = df.dropna(subset=["rel_strike", "market_iv"]).groupby(["date", "maturity"])
    for (date, mat), group in grouped:
        strikes = group["rel_strike"].values
        vols = group["market_iv"].values
        try:
            iv_80 = vols[np.isclose(strikes, target_strikes[0])][0]
            iv_100 = vols[np.isclose(strikes, target_strikes[1])][0]
            iv_120 = vols[np.isclose(strikes, target_strikes[2])][0]
            skew = iv_80 - iv_120
            convexity = 2 * iv_100 - iv_80 - iv_120
            rows.append((date, mat, skew, convexity))
        except IndexError:
            continue
    return pd.DataFrame(rows, columns=["date", "maturity", "smile_skew", "smile_convexity"]).set_index(["date", "maturity"])


def compute_term_structure_features(df, target_strike=1.0, tau_map=None):
    """Computes term structure features at ATM strike, including slopes and averages."""
    rows = []
    grouped = df[df["rel_strike"].between(0.99, 1.01)].dropna(subset=["market_iv"]).groupby("date")
    for date, group in grouped:
        if tau_map:
            group = group.assign(tau=group["maturity"].map(tau_map))
        group = group.sort_values("tau")
        taus = group["tau"].values
        ivs = group["market_iv"].values
        if len(ivs) < 5:
            continue
        short_mask = taus <= 1.0
        long_mask = taus > 1.0
        level = np.mean(ivs)
        slope = ivs[-1] - ivs[0]
        mid_idx = len(ivs) // 2
        curvature = ivs[mid_idx] - 0.5 * (ivs[0] + ivs[-1])
        tau_weighted = np.sum(ivs * taus) / (np.sum(taus) + 1e-6)
        short_avg = np.mean(ivs[short_mask]) if np.any(short_mask) else np.nan
        long_avg = np.mean(ivs[long_mask]) if np.any(long_mask) else np.nan
        slope_split = long_avg - short_avg if np.isfinite(short_avg) and np.isfinite(long_avg) else np.nan
        rows.append((
            date, level, slope, curvature,
            tau_weighted, short_avg, long_avg, slope_split
        ))
    return pd.DataFrame(
        rows,
        columns=[
            "date", "ts_level", "ts_slope", "ts_curvature",
            "ts_weighted", "ts_short_avg", "ts_long_avg", "ts_split_slope"
        ]
    ).set_index("date")


