import numpy as np
import pandas as pd
from keras import ops
from scipy.interpolate import interp1d

def interpolate_rate(row, tau_target):
    """Interpolates rate for target tau using available rf_ columns."""
    rate_cols = [col for col in row.index if col.startswith('rf_') and pd.notna(row[col])]
    if len(rate_cols) < 2:
        return np.nan
    tau_rate_pairs = sorted(
        [(float(col.replace("rf_", "")), row[col]) for col in rate_cols],
        key=lambda x: x[0]
    )
    taus, rates = zip(*tau_rate_pairs)
    return float(interp1d(taus, rates, kind='linear', fill_value='extrapolate')(tau_target))


def build_vol_tensors(df, maturity_to_tau):
    """Builds tensors for volatility surfaces from DataFrame.
    Key tensor dimensions:
        date_tensor: (T,)
        spot_tensor: (T,)
        curve_tensor: (T, 32)
        surface_tensor: (T, M, K, 6)
        ssvi_tensor: (T, M, K, 3)
        heston_tensor: (T, M, 5)
        strike_tensor: (K,)
        abs_strike_tensor: (T, K)
    T: number of dates, M: number of maturities, K: number of strikes
    """
    df_clean = df.dropna(subset=[col for col in df.columns if col.startswith('rf_')])
    df_clean = df_clean.sort_values(["date", "maturity", "rel_strike", "tau"]).copy()

    dates = sorted(df_clean['date'].unique())
    maturities = sorted(maturity_to_tau.keys(), key=lambda m: maturity_to_tau[m])
    taus = [maturity_to_tau[m] for m in maturities]
    rel_strikes = sorted(df_clean['rel_strike'].unique())

    n_dates, n_maturities, n_strikes = len(dates), len(maturities), len(rel_strikes)
    date_index = {d: i for i, d in enumerate(dates)}
    mat_index = {m: i for i, m in enumerate(maturities)}
    strike_index = {k: i for i, k in enumerate(rel_strikes)}

    date_tensor = np.array(dates)  # (T,)
    spot_tensor = np.full((n_dates,), np.nan, dtype=np.float32)  # (T,)
    curve_tensor = np.full((n_dates, 32), np.nan, dtype=np.float32)  # (T, 32)
    surface_tensor = np.full((n_dates, n_maturities, n_strikes, 6), np.nan, dtype=np.float32)  # (T, M, K, 6)
    ssvi_tensor = np.full((n_dates, n_maturities, n_strikes, 3), np.nan, dtype=np.float32)  # (T, M, K, 3)
    heston_tensor = np.full((n_dates, n_maturities, 5), np.nan, dtype=np.float32)  # (T, M, 5)

    grouped = df_clean.groupby(['date', 'maturity'])
    for (date, mat), group in grouped:
        if mat not in maturity_to_tau: continue
        tau = maturity_to_tau[mat]
        d_idx = date_index[date]
        m_idx = mat_index[mat]
        group = group.sort_values("rel_strike")

        if np.isnan(spot_tensor[d_idx]):
            spot_tensor[d_idx] = group['S0'].iloc[0]
            rf_cols = sorted([c for c in df.columns if c.startswith("rf_")], key=lambda x: float(x.replace("rf_", "")))
            curve_tensor[d_idx] = group[rf_cols].iloc[0].to_numpy(dtype=np.float32)

        heston_values = group[['heston_kappa', 'heston_theta', 'heston_sigma', 'heston_rho', 'heston_v0']].iloc[0].to_numpy()
        heston_values = np.where(heston_values == 0, np.nan, heston_values)
        if np.isnan(heston_tensor[d_idx, m_idx, 0]):
            heston_tensor[d_idx, m_idx] = heston_values

        for _, row in group.iterrows():
            rel_k = round(row['rel_strike'], 3)
            if rel_k not in strike_index: continue
            r_interp = interpolate_rate(row, tau)
            if not np.isfinite(r_interp): continue
            k_idx = strike_index[rel_k]
            surface_tensor[d_idx, m_idx, k_idx] = [
                row['S0'], row['K'], tau, row['q'], r_interp, row['market_iv']
            ]
            ssvi_tensor[d_idx, m_idx, k_idx] = [
                row['ssvi_theta'], row['ssvi_rho'], row['ssvi_beta']
            ]

    strike_tensor = ops.convert_to_tensor(np.array(rel_strikes, dtype=np.float32))  # (K,)
    abs_strike_tensor = ops.expand_dims(spot_tensor, axis=1) * strike_tensor  # (T, K)

    return {
        "date_tensor": date_tensor,  # (T,)
        "spot_tensor": ops.convert_to_tensor(spot_tensor),  # (T,)
        "curve_tensor": ops.convert_to_tensor(curve_tensor),  # (T, 32)
        "surface_tensor": ops.convert_to_tensor(surface_tensor),  # (T, M, K, 6)
        "ssvi_tensor": ops.convert_to_tensor(ssvi_tensor),  # (T, M, K, 3)
        "heston_tensor": ops.convert_to_tensor(heston_tensor),  # (T, M, 5)
        "strike_tensor": strike_tensor,  # (K,)
        "abs_strike_tensor": abs_strike_tensor,  # (T, K)
        "dates": dates,
        "maturities": maturities,
        "taus": taus,
        "rel_strikes": rel_strikes,
        "date_index": date_index,
        "maturity_to_tau": maturity_to_tau
    }

def split_time_series_indices(n, train=0.7, val=0.15):
    """Splits time series indices into train, val, test sets."""
    t_train = int(n * train)
    t_val = int(n * (train + val))
    return np.arange(0, t_train), np.arange(t_train, t_val), np.arange(t_val, n)

def slice_tensors(tensors, indices):
    """Slices tensors by given indices."""
    return dict(
        surface_tensor=tensors["surface_tensor"][indices],
        spot_tensor=tensors["spot_tensor"][indices],
        curve_tensor=tensors["curve_tensor"][indices],
        date_tensor=tensors["date_tensor"][indices] if isinstance(tensors["date_tensor"], np.ndarray)
        else np.array(tensors["date_tensor"])[indices]
    )
def preprocess_tau(tau):
    """Preprocesses tau values for feature engineering."""
    tau = np.array(tau, dtype="float32").reshape(-1, 1)
    tau = np.clip(tau, 1e-4, None)
    return np.concatenate([
        tau,
        np.log(tau),
        1 / np.sqrt(tau),
        (tau - np.mean(tau)) / (np.std(tau) + 1e-4)
    ], axis=1)

from data.feature_engineering import interpolate_monthly_maturities
from keras import ops

def prepare_surface_and_feature_tensors(
    df_all,
    df_raw,
    feat_cols,
    monthly_interpolation=True
):
    """Prepares surface and feature tensors, optionally with monthly interpolation.
    Key tensor dimensions:
        surface_tensor: (T, M, K, 6)
        iv_diff_tensor: (T, M, K, 7)
        X_feat_tensor: (T, F + M*K)
    """
    maturity_to_tau = {
        '1m': 1/12, '2m': 2/12, '3m': 3/12, '6m': 0.5, '9m': 0.75,
        '1y': 1.0, '18m': 1.5, '2y': 2.0, '3y': 3.0, '4y': 4.0, '5y': 5.0
    }
    df_aligned = df_raw[df_raw["date"].isin(df_all.index)].copy()
    tensors = build_vol_tensors(df_aligned, maturity_to_tau)
    if monthly_interpolation:
        monthly_maturities = [1/12 * i for i in range(1, 61)]
        interpolated_surface_tensor, updated_taus = interpolate_monthly_maturities(
            tensors["surface_tensor"],
            tensors["taus"],
            monthly_maturities
        )
        tensors["surface_tensor"] = interpolated_surface_tensor
        tensors["taus"] = updated_taus
    aligned_dates = list(tensors["date_tensor"])
    df_feat = df_all.loc[aligned_dates, feat_cols]
    X_feat_tensor = ops.convert_to_tensor(df_feat.values.astype("float32"))
    tensors["iv_diff_tensor"] = add_iv_differences(tensors["surface_tensor"])
    iv_diff_tensor = ops.convert_to_numpy(tensors["iv_diff_tensor"][..., -1])  # (T, M, K)
    iv_diff_flat = iv_diff_tensor.reshape(iv_diff_tensor.shape[0], -1)         # (T, M*K)
    X_feat_tensor = ops.concatenate([X_feat_tensor, ops.convert_to_tensor(iv_diff_flat.astype("float32"))], axis=-1)
    return tensors, X_feat_tensor, df_feat


## not used for now , but can be used as a surface input fro the model 
def add_iv_differences(surface_tensor):
    """Adds first difference of implied volatility as feature channel.
    Input: surface_tensor (T, M, K, C)
    Output: (T, M, K, C+1) with IV_diff appended
    """
    surface_np = ops.convert_to_numpy(surface_tensor)
    T, M, K, C = surface_np.shape
    iv = surface_np[..., -1]
    iv_diff = np.zeros_like(iv)
    iv_diff[1:] = iv[1:] - iv[:-1]
    iv_diff[0] = iv[0]
    iv_diff = iv_diff[..., np.newaxis]
    combined = np.concatenate([surface_np, iv_diff], axis=-1)
    return ops.convert_to_tensor(combined.astype("float32"))
