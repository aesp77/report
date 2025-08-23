import datetime as dt
from psccommon.apiclient.PscMarqueeClient import PscMarqueeClient 
import pandas as pd
from psccommon.db.db import get_db_async
import asyncio
import nest_asyncio
from xbbg import blp
from datetime import datetime
from pathlib import Path

def load_iv_dataset(path=None):
    """Loads IV dataset from CSV file."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / 'data' / 'SPX_Index_history_dataset.csv'
    return pd.read_csv(path, parse_dates=['date']).sort_values('date').set_index('date')
# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

#---- Fetch data from proprietary database ---#


async def get_marquee_id(stock_ticker: str) -> str:
    """Fetches Marquee ID for a given stock ticker from database."""
    normalized_ticker = stock_ticker.split()[0]
    db = await get_db_async('Prod')
    sql = "SELECT id, bbid FROM tbl_psc_static_gs"
    df = pd.DataFrame([dict(r) for r in await db.fetch(sql)])
    result = df[df['bbid'].str.contains(normalized_ticker, case=False, na=False)]
    if result.empty:
        raise ValueError(f"No Marquee ID found for ticker: {stock_ticker}.")
    marquee_id = result.iloc[0]['id']
    return marquee_id

async def get_spot_data(ticker: str, begin: str, end: str) -> pd.DataFrame:
    """Fetches spot price data and computes daily returns."""
    db = await get_db_async('Prod')
    sql = f"""
        SELECT date, last FROM tbl_psc_market_data_spot_prices
        WHERE ticker = '{ticker}' AND date BETWEEN '{begin}' AND '{end}'
    """
    df = pd.DataFrame([dict(r) for r in await db.fetch(sql)])
    df.sort_values('date', inplace=True)
    df['daily_return'] = df['last'].pct_change()
    return df.dropna()

async def get_vol_surface(marquee_id: str, begin: str, end: str) -> pd.DataFrame:
    """Fetches implied volatility surface data from Marquee API."""
    m = PscMarqueeClient()
    strikes = [i / 100 for i in range(60, 160, 10)]
    tenors = ['1m', '2m', '3m', '6m', '9m', '1y', '18m', '2y', '3y', '4y', '5y', '6y', '7y']
    all_data = []
    for strike, tenor in [(s, t) for s in strikes for t in tenors]:
        data = await m.get_data(
            'EDRVOL_PERCENT_PREMIUM_PTS_EXT',
            asset_id=marquee_id,
            start_date=begin,
            end_date=end,
            extra_params={
                'tenor': tenor,
                'relativeStrike': strike,
                'strikeReference': 'forward'
            }
        )
        if data:
            df = pd.DataFrame(data).drop(columns=['assetId', 'strikeReference', 'absoluteStrike', 'updateTime'])
            df['strike'] = strike
            df['tenor'] = tenor
            all_data.append(df)
    return pd.concat(all_data, axis=0)

async def get_ssvi_parameters(ticker: str, begin: str, end: str) -> pd.DataFrame:
    """Fetches SSVI parameters from database."""
    db = await get_db_async("Prod")
    sql = f"""
        SELECT cob_date, theta, rho, beta 
        FROM tbl_market_data_ssvi_parameters_eod 
        WHERE underlying = '{ticker}' 
        AND cob_date BETWEEN '{begin}' AND '{end}'
    """
    data = await db.fetch(sql)
    df = pd.DataFrame([dict(d) for d in data])
    df.rename(columns={'cob_date': 'date'}, inplace=True)
    df.sort_values('date', inplace=True)
    return df

def get_dividend_yield(ticker: str, begin: str, end: str) -> pd.DataFrame:
    """Fetches daily dividend yield from Bloomberg."""
    try:
        df = blp.bdh(
            tickers=ticker,
            flds='BEST_DIV_YLD',
            start_date=begin,
            end_date=end,
            Per='D'
        )
        df.columns = ['dividend_yield']
        df = df.reset_index().rename(columns={'index': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'dividend_yield']]
    except Exception:
        return pd.DataFrame()

def get_ticker_year_fractions(tickers: list) -> dict:
    """Computes year fractions to maturity for tickers."""
    today = pd.Timestamp(datetime.today().date())
    meta = blp.bdp(tickers=tickers, flds='MATURITY')
    meta = meta.dropna()
    ticker_to_tau = {}
    for t in meta.index:
        mat_date = pd.to_datetime(meta.loc[t, 'maturity'])
        tau = (mat_date - today).days / 365.0
        if tau > 0:
            ticker_to_tau[t] = f"rf_{round(tau, 6)}"
    return ticker_to_tau

def get_ois_curve(currency: str, begin: str, end: str) -> pd.DataFrame:
    """Fetches OIS curve data for a currency and date range from Bloomberg."""
    tickers = [
        "SOFRRATE INDEX", "USOSFR1Z Curncy", "USOSFR2Z Curncy", "USOSFR3Z Curncy", "USOSFRA Curncy",
        "USOSFRB Curncy", "USOSFRC Curncy", "USOSFRD Curncy", "USOSFRE Curncy", "USOSFRF Curncy",
        "USOSFRG Curncy", "USOSFRH Curncy", "USOSFRI Curncy", "USOSFRJ Curncy", "USOSFRK Curncy",
        "USOSFR1 Curncy", "USOSFR1F Curncy", "USOSFR2 Curncy", "USOSFR3 Curncy", "USOSFR4 Curncy",
        "USOSFR5 Curncy", "USOSFR6 Curncy", "USOSFR7 Curncy", "USOSFR8 Curncy", "USOSFR9 Curncy",
        "USOSFR10 Curncy", "USOSFR12 Curncy", "USOSFR15 Curncy", "USOSFR20 Curncy", "USOSFR25 Curncy",
        "USOSFR30 Curncy", "USOSFR40 Curncy", "USOSFR50 Curncy"
    ]

    df = blp.bdh(tickers=tickers, flds='PX_LAST', start_date=begin, end_date=end)
    if df.empty:
        return pd.DataFrame()

    ticker_to_tau = get_ticker_year_fractions(tickers)
    if not ticker_to_tau:
        return pd.DataFrame()

    df.columns = [ticker_to_tau.get(col[0], None) for col in df.columns]
    df = df.loc[:, ~pd.isna(df.columns)]
    df.columns.name = None

    df.index.name = "date"
    df.reset_index(inplace=True)
    return df

#---- Mergin called datasets ---#

def merge_datasets(spot_df, div_df, ois_df, ssvi_df, vol_pivot):
    """Merges spot, dividend, OIS, SSVI, and volatility datasets by date."""
    if not div_df.empty:
        div_df["date"] = pd.to_datetime(div_df["date"])
        spot_df["date"] = pd.to_datetime(spot_df["date"])
        merged = pd.merge_asof(spot_df.sort_values("date"), div_df.sort_values("date"), on="date", direction="backward")
    else:
        merged = spot_df.copy()
    if not ois_df.empty:
        ois_df["date"] = pd.to_datetime(ois_df["date"])
        merged["date"] = pd.to_datetime(merged["date"])
        merged = pd.merge_asof(merged.sort_values("date"), ois_df.sort_values("date"), on="date", direction="backward")
    if not ssvi_df.empty:
        ssvi_df["date"] = pd.to_datetime(ssvi_df["date"])
        merged = merged.merge(ssvi_df, on="date", how="inner")
    if not vol_pivot.empty:
        vol_pivot["date"] = pd.to_datetime(vol_pivot["date"])
        merged = merged.merge(vol_pivot, on="date", how="left")
    merged.sort_values("date", inplace=True)
    return merged

class DatasetCreator:
    """Creates and saves merged dataset from multiple sources."""
    def __init__(self, ticker, begin, end, currency):
        self.ticker = ticker
        self.begin = begin
        self.end = end
        self.currency = currency

    def create_and_save_dataset(self, output_path):
        spot_df = asyncio.run(get_spot_data(self.ticker, self.begin, self.end))
        ssvi_df = asyncio.run(get_ssvi_parameters(self.ticker, self.begin, self.end))
        ssvi_df = ssvi_df.groupby("date")[["theta", "rho", "beta"]].mean().reset_index()
        marquee_id = asyncio.run(get_marquee_id(self.ticker))
        vol_raw_df = asyncio.run(get_vol_surface(marquee_id, self.begin, self.end))
        vol_pivot = vol_raw_df.pivot_table(index="date", columns=["strike", "tenor"], values="impliedVolatility")
        vol_pivot.columns = [f"iv_{int(s*100)}_{t}" for s, t in vol_pivot.columns]
        vol_pivot.reset_index(inplace=True)
        div_df = get_dividend_yield(self.ticker, self.begin, self.end)
        ois_df = get_ois_curve(self.currency, self.begin, self.end)
        merged = merge_datasets(spot_df, div_df, ois_df, ssvi_df, vol_pivot)
        merged.to_csv(output_path, index=False)

def load_and_clean_raw_dataset(filepath: str):
    """Loads and cleans raw dataset from CSV file."""
    import pandas as pd
    import numpy as np
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.weekday < 5]
    df["rel_strike"] = (df["K"] / df["S0"]).round(2)
    expected_strikes = np.round(np.arange(0.6, 1.51, 0.1), 2).tolist()
    df = df[df["rel_strike"].isin(expected_strikes)].copy()
    df = df.sort_values(["date", "rel_strike", "tau"]).reset_index(drop=True)
    return df
