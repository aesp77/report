import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ssvi fitting and reconstruction functions
def fit_ssvi_params(Ks, ivs, tau):
    log_m = np.log(Ks / Ks.mean())
    target_var = (ivs ** 2) * tau

    def ssvi_w(log_m, theta, rho, beta):
        phi = beta / (tau * theta)
        return theta / 2 * (1 + rho * phi * log_m + np.sqrt((phi * log_m + rho)**2 + 1 - rho**2))

    def objective(params):
        theta, rho, beta = params
        if theta <= 0 or not (-0.999 < rho < 0.999) or beta <= 0:
            return np.inf
        w_model = ssvi_w(log_m, theta, rho, beta)
        return np.mean((target_var - w_model) ** 2)

    res = minimize(objective, x0=[0.1, -0.3, 5.0],
                   bounds=[(1e-6, 5), (-0.999, 0.999), (1e-6, 100)],
                   method='L-BFGS-B')
    return res.x if res.success else (np.nan, np.nan, np.nan)


def extract_ssvi_surface(df, maturity_map, maturity_to_tau):
    rows = []

    for date, row in df.iterrows():
        S0 = row['last']
        for maturity, cols in maturity_map.items():
            tau = maturity_to_tau.get(maturity)
            if tau is None:
                continue

            valid = [(col, strike) for col, strike in cols if pd.notna(row[col])]
            if len(valid) < 5:
                continue

            ivs = np.array([row[col] for col, _ in valid])
            Ks = np.array([S0 * strike / 100 for _, strike in valid])

            theta, rho, beta = fit_ssvi_params(Ks, ivs, tau)
            rows.append(dict(date=date, maturity=maturity, theta=theta, rho=rho, beta=beta))

    return pd.DataFrame(rows).set_index(['date', 'maturity'])


def ssvi_iv(log_m, tau, theta, rho, beta):
    phi = beta / (tau * theta)
    w = theta / 2 * (1 + rho * phi * log_m + np.sqrt((phi * log_m + rho)**2 + 1 - rho**2))
    return np.sqrt(w / tau)


def plot_ssvi_reconstruction(df, ssvi_df, maturity_map, maturity_to_tau, interpolate_rate, sample_date, maturity):
    row = df.loc[sample_date]
    spot = row['last']
    tau = maturity_to_tau[maturity]

    Ks, market_IVs = zip(*[
        (strike / 100 * spot, row[col])
        for col, strike in maturity_map[maturity]
        if pd.notna(row[col])
    ])
    Ks = np.array(Ks)
    market_IVs = np.array(market_IVs)

    q = row['dividend_yield'] / 100
    r = interpolate_rate(row, tau)
    log_m = np.log(Ks / (spot * np.exp((r - q) * tau)))

    theta, rho, beta = ssvi_df.loc[(sample_date, maturity)].values
    ssvi_IVs = ssvi_iv(log_m, tau, theta, rho, beta)

    plt.figure(figsize=(8, 6))
    plt.plot(Ks / spot, market_IVs, label="Market IV", marker='o')
    plt.plot(Ks / spot, ssvi_IVs, label="SSVI IV (fitted)", marker='x')
    plt.title(f"SSVI Surface Fit — {sample_date}, {maturity}")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()
    plt.show()