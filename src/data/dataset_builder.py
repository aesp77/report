import pandas as pd
import numpy as np

from data.loader import load_and_clean_raw_dataset
from data.feature_engineering import (
    compute_log_returns, compute_realized_vol,
    compress_yield_curve, compute_rolling_features,
    compute_smile_features_by_maturity, compute_term_structure_features
)

def build_full_surface_feature_df(df_path: str, dropna: bool = True) -> pd.DataFrame:
    """
    build feature dataframe for full volatility surface
    combines surface, curve, and engineered features
    input: df_path (str), dropna (bool)
    output: dataframe indexed by date
    """
    df = load_and_clean_raw_dataset(df_path)
    maturity_to_tau = {
        '1m': 1/12, '2m': 2/12, '3m': 3/12, '6m': 0.5, '9m': 0.75,
        '1y': 1.0, '18m': 1.5, '2y': 2.0, '3y': 3.0, '4y': 4.0, '5y': 5.0
    }
    # filter and sort surface data
    df = df.dropna(subset=["market_iv", "rel_strike", "maturity"])
    df = df.sort_values(["date", "maturity", "rel_strike"])
    dates = sorted(df["date"].unique())
    rel_strikes = sorted(df["rel_strike"].unique())
    maturities = sorted(maturity_to_tau.keys(), key=lambda m: maturity_to_tau[m])
    # pivot surface to (date, maturity, strike)
    pivot_df = df.pivot_table(
        index="date",
        columns=["maturity", "rel_strike"],
        values="market_iv"
    ).reindex(index=dates, columns=pd.MultiIndex.from_product([maturities, rel_strikes]))
    if dropna:
        pivot_df = pivot_df.dropna()
    # flatten column names
    flat_cols = [f"iv_{mat}_{strike:.2f}" for mat, strike in pivot_df.columns]
    surface_df = pivot_df.copy()
    surface_df.columns = flat_cols
    # build feature block
    rf_cols = [col for col in df.columns if col.startswith("rf_")]
    daily_df = df.sort_values(["date", "maturity"]).groupby("date").first()
    returns = compute_log_returns(daily_df)
    realized_vol = compute_realized_vol(returns, window=5)
    r_ma, r_z = compute_rolling_features(returns)
    v_ma, v_z = compute_rolling_features(realized_vol)
    curve_features = compress_yield_curve(daily_df, rf_cols)
    smile_by_maturity = compute_smile_features_by_maturity(df)
    smile_features = smile_by_maturity.groupby("date").mean()
    term_features = compute_term_structure_features(df, tau_map=maturity_to_tau)
    features_df = pd.concat(
        [returns, r_ma, r_z, realized_vol, v_ma, v_z, curve_features], axis=1
    ).join([smile_features, term_features], how="inner")
    # align feature and surface blocks on common dates
    common_dates = features_df.index.intersection(surface_df.index)
    surface_df = surface_df.loc[common_dates]
    features_df = features_df.loc[common_dates]
    # combine into final dataframe
    df_final = pd.concat([surface_df, features_df], axis=1)
    df_final.index.name = "date"
    return df_final


#-------- DO NOT USE , SSVI ANALYSIS NOT COMPLETED 

def build_ssvi_feature_df(df_path: str, maturity_to_tau: dict = None, dropna: bool = True) -> pd.DataFrame:
    """
    build feature dataframe including ssvi parameters
    combines curve, engineered features, and ssvi block
    input: df_path (str), maturity_to_tau (dict), dropna (bool)
    output: dataframe indexed by date
    """
    if maturity_to_tau is None:
        maturity_to_tau = {
            '1m': 1/12, '2m': 2/12, '3m': 3/12, '6m': 0.5, '9m': 0.75,
            '1y': 1.0, '18m': 1.5, '2y': 2.0, '3y': 3.0, '4y': 4.0, '5y': 5.0
        }
    df = load_and_clean_raw_dataset(df_path)
    df = df.dropna(subset=["ssvi_theta", "ssvi_beta", "ssvi_rho"])
    # build surface-wide feature block
    rf_cols = [col for col in df.columns if col.startswith("rf_")]
    daily_df = df.sort_values(["date", "maturity"]).groupby("date").first()
    returns = compute_log_returns(daily_df)
    realized_vol = compute_realized_vol(returns, window=5)
    r_ma, r_z = compute_rolling_features(returns)
    v_ma, v_z = compute_rolling_features(realized_vol)
    curve_features = compress_yield_curve(daily_df, rf_cols)
    smile_by_maturity = compute_smile_features_by_maturity(df)
    smile_features = smile_by_maturity.groupby("date").mean()
    term_features = compute_term_structure_features(df, tau_map=maturity_to_tau)
    features_df = pd.concat(
        [returns, r_ma, r_z, realized_vol, v_ma, v_z, curve_features], axis=1
    ).join([smile_features, term_features], how="inner")
    # pivot ssvi parameters into (date, maturity) format
    ssvi_df = df.dropna(subset=["ssvi_theta", "ssvi_beta", "ssvi_rho"])
    ssvi_group = ssvi_df.groupby(["date", "maturity"])[["ssvi_theta", "ssvi_beta", "ssvi_rho"]].first().unstack("maturity")
    # flatten ssvi column names
    ssvi_group.columns = [f"{param}_{mat}" for param, mat in ssvi_group.columns]
    if dropna:
        ssvi_group = ssvi_group.dropna()
    # align feature and ssvi blocks on common dates
    common_dates = features_df.index.intersection(ssvi_group.index)
    features_df = features_df.loc[common_dates]
    ssvi_group = ssvi_group.loc[common_dates]
    # combine into final dataframe
    df_final = pd.concat([features_df, ssvi_group], axis=1)
    df_final.index.name = "date"
    return df_final
