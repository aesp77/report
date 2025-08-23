# edav.py - Enhanced for Volatility Surface Pipeline Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import keras
from keras import ops

# ============= Existing Functions (Cleaned) =============

def quick_overview(df: pd.DataFrame):
    """
    quick visual overview of dataframe: boxplot for numerical columns and time series for 'last' column
    input: df (pandas dataframe)
    output: matplotlib plots
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols.drop('last', errors='ignore')
    if not numerical_cols.empty:
        sns.boxplot(data=df[numerical_cols], ax=axes[0])
        axes[0].set_title("Boxplot of Numerical Columns")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].fontsize = 4

    if 'last' in df.columns:
        axes[1].plot(df.index, df['last'], label='Last Data', linewidth=1)
        axes[1].set_title("Last Data Time Series")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Last Value")
        axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_dataset(df: pd.DataFrame):
    """
    pairplot and correlation heatmap for dataframe
    input: df (pandas dataframe)
    output: matplotlib plots
    """
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 1:
        sns.pairplot(df[numerical_cols])
        plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def analyze_dataset(df: pd.DataFrame):
    """
    returns basic stats: row/col count, missing, duplicates, unique values
    input: df (pandas dataframe)
    output: dict of stats
    """
    stats = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing': df.isnull().sum()[df.isnull().sum() > 0],
        'duplicates': df.duplicated().sum(),
        'unique_values': df.nunique()
    }
    return stats

# ============= Surface Analysis Functions =============

def analyze_surface_completeness(df, date_col='date', maturity_col='maturity', strike_col='rel_strike'):
    """
    completeness and missing patterns for surface data
    input: df (surface dataframe), date/maturity/strike column names
    output: dict with completeness stats and missing heatmap
    """
    completeness = df.groupby([date_col]).apply(
        lambda x: len(x) / (df[maturity_col].nunique() * df[strike_col].nunique())
    )
    
    missing_patterns = df.groupby([maturity_col, strike_col]).size().unstack(fill_value=0)
    
    return {
        'daily_completeness': completeness,
        'mean_completeness': completeness.mean(),
        'missing_heatmap': missing_patterns,
        'worst_days': completeness.nsmallest(10)
    }

def detect_iv_outliers(df, iv_col='market_iv', method='iqr', threshold=3):
    """
    detect outliers in implied volatility using zscore or iqr method
    input: df (surface dataframe), iv_col, method, threshold
    output: dict with outlier mask/count/pct
    """
    results = {}
    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df[iv_col].dropna()))
        outliers = z_scores > threshold
        results['outlier_mask'] = outliers
        results['outlier_count'] = outliers.sum()
        
    elif method == 'iqr':
        Q1 = df[iv_col].quantile(0.25)
        Q3 = df[iv_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[iv_col] < (Q1 - 1.5 * IQR)) | (df[iv_col] > (Q3 + 1.5 * IQR))
        results['outlier_mask'] = outliers
        results['outlier_count'] = outliers.sum()
    
    results['outlier_pct'] = results['outlier_count'] / len(df) * 100
    return results

def analyze_temporal_gaps(df, date_col='date'):
    """
    analyze gaps in time series dates
    input: df (surface dataframe), date_col
    output: dict with gap stats and missing dates
    """
    df_sorted = df.sort_values(date_col)
    dates = pd.Series(pd.to_datetime(df_sorted[date_col].unique())).sort_values()
    
    gaps = pd.Series(dates).diff()
    business_days = pd.bdate_range(dates.min(), dates.max())
    missing_days = business_days.difference(dates)
    
    return {
        'total_days': len(dates),
        'expected_business_days': len(business_days),
        'missing_business_days': len(missing_days),
        'largest_gap': gaps.max(),
        'gap_distribution': gaps.value_counts(),
        'missing_dates': missing_days
    }

# ============= Tensor-specific Analysis =============

def analyze_temporal_gaps_tensor(date_tensor):
    """
    analyze gaps in tensor date sequence
    input: date_tensor (array-like)
    output: dict with gap summary and chronological check
    """
    dates = pd.to_datetime(date_tensor)
    df_dates = pd.DataFrame({'date': dates}).sort_values('date')
    
    df_dates['gap_days'] = df_dates['date'].diff().dt.days
    
    # Business days only - no weekends in data
    df_dates['gap_type'] = 'normal'
    df_dates.loc[df_dates['gap_days'] == 1, 'gap_type'] = 'consecutive'
    df_dates.loc[df_dates['gap_days'] > 1, 'gap_type'] = 'holiday/missing'
    
    return {
        'gap_summary': df_dates['gap_type'].value_counts(),
        'max_gap': df_dates['gap_days'].max(),
        'gaps_over_1': (df_dates['gap_days'] > 1).sum(),
        'chronological': df_dates['date'].is_monotonic_increasing
    }

def compute_surface_autocorrelation(surface_tensor):
    """
    compute autocorrelation grid for volatility surface tensor
    input: surface_tensor (t, m, k, c)
    output: dict with autocorr grid and summary stats
    """
    T, M, K, C = surface_tensor.shape
    iv_surfaces = ops.convert_to_numpy(surface_tensor[:, :, :, -1])
    
    autocorr_grid = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            series = iv_surfaces[:, m, k]
            if np.std(series) > 0:
                autocorr_grid[m, k] = np.corrcoef(series[:-1], series[1:])[0, 1]
    
    return {
        'autocorr_grid': autocorr_grid,
        'mean_autocorr': np.mean(autocorr_grid),
        'min_autocorr': np.min(autocorr_grid)
    }

def analyze_surface_tensor_statistics(surface_tensor, taus, strikes):
    """
    compute time series and maturity profile stats for surface tensor
    input: surface_tensor (t, m, k, c), taus, strikes
    output: dict with time series stats and maturity profiles
    """
    T, M, K, C = surface_tensor.shape
    iv_surfaces = ops.convert_to_numpy(surface_tensor[:, :, :, -1])
    
    # Surface statistics over time
    surface_stats = []
    for t in range(T):
        surf = iv_surfaces[t]
        stats = {
            'mean_iv': np.mean(surf),
            'std_iv': np.std(surf),
            'atm_iv': surf[M//2, K//2],
            'skew': np.mean(surf[:, :K//2]) - np.mean(surf[:, K//2:]),  # Put-call skew
            'term_slope': surf[-1, K//2] - surf[0, K//2],  # Term structure slope at ATM
            'smile_curvature': 2*surf[M//2, K//2] - surf[M//2, 0] - surf[M//2, -1]  # ATM smile curvature
        }
        surface_stats.append(stats)
    
    df_stats = pd.DataFrame(surface_stats)
    
    # Maturity-specific patterns
    maturity_profiles = {}
    for m in range(M):
        maturity_profiles[f'tau_{taus[m]:.2f}'] = {
            'mean_iv': np.mean(iv_surfaces[:, m, :]),
            'vol_of_vol': np.std(iv_surfaces[:, m, :]),
            'smile_width': np.mean(np.abs(iv_surfaces[:, m, :] - iv_surfaces[:, m, K//2:K//2+1]))
        }
    
    return {
        'time_series_stats': df_stats,
        'maturity_profiles': pd.DataFrame(maturity_profiles).T,
        'overall_mean': np.mean(iv_surfaces),
        'overall_std': np.std(iv_surfaces)
    }

def visualize_surface_characteristics(surface_tensor, taus, strikes, dates):
    """
    plot key surface characteristics over time and by maturity
    input: surface_tensor, taus, strikes, dates
    output: matplotlib plots, returns time series stats dataframe
    """
    analysis = analyze_surface_tensor_statistics(surface_tensor, taus, strikes)
    df_stats = analysis['time_series_stats']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # ATM IV time series
    axes[0, 0].plot(dates, df_stats['atm_iv'], linewidth=0.5)
    axes[0, 0].set_title('ATM IV Over Time')
    axes[0, 0].set_ylabel('IV')
    
    # Term structure slope
    axes[0, 1].plot(dates, df_stats['term_slope'], linewidth=0.5, color='orange')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Term Structure Slope (ATM)')
    axes[0, 1].set_ylabel('Long - Short IV')
    
    # Put-Call Skew
    axes[0, 2].plot(dates, df_stats['skew'], linewidth=0.5, color='green')
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_title('Lower Strike-Upper Strike Skew')
    axes[0, 2].set_ylabel('Lower Strike IV - Upper Strike IV')

    # Rolling volatility of ATM
    rolling_std = df_stats['atm_iv'].rolling(20).std()
    axes[1, 0].plot(dates, rolling_std, linewidth=0.5, color='red')
    axes[1, 0].set_title('20-Day Vol of ATM IV')
    axes[1, 0].set_ylabel('Rolling Std')
    
    # Maturity profiles
    maturity_data = analysis['maturity_profiles']
    axes[1, 1].bar(range(len(maturity_data)), maturity_data['mean_iv'])
    axes[1, 1].set_xticks(range(len(maturity_data)))
    axes[1, 1].set_xticklabels([f'{t:.2f}' for t in taus], rotation=45)
    axes[1, 1].set_title('Mean IV by Maturity')
    axes[1, 1].set_xlabel('Tau')
    axes[1, 1].set_ylabel('Mean IV')
    
    # Smile curvature distribution
    axes[1, 2].hist(df_stats['smile_curvature'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(x=0, color='r', linestyle='--')
    axes[1, 2].set_title('Smile Curvature Distribution')
    axes[1, 2].set_xlabel('Curvature')
    
    plt.tight_layout()
    plt.show()
    
    return df_stats

# ============= Calendar Arbitrage Validation =============

def check_calendar_arbitrage(surface_tensor, taus, tolerance=1e-6):
    """
    check calendar arbitrage violations in surface tensor
    input: surface_tensor, taus, tolerance
    output: dict with violation stats
    """
    violations = []
    total_var = surface_tensor**2 * taus.reshape(-1, 1)
    
    for i in range(len(taus) - 1):
        violation_mask = total_var[i+1] < total_var[i] - tolerance
        if violation_mask.any():
            violations.append({
                'tau_pair': (taus[i], taus[i+1]),
                'violation_count': violation_mask.sum(),
                'violation_pct': violation_mask.mean() * 100,
                'max_violation': (total_var[i] - total_var[i+1])[violation_mask].max()
            })
    
    return {
        'total_violations': sum(v['violation_count'] for v in violations),
        'violation_pairs': violations,
        'clean_surfaces_pct': 100 - (len(violations) / (len(taus)-1) * 100)
    }

def check_butterfly_arbitrage(surface, strikes, tolerance=1e-6):
    """
    check butterfly arbitrage violations in surface
    input: surface (m, k), strikes, tolerance
    output: dict with violation stats
    """
    violations = 0
    for i in range(1, len(strikes) - 1):
        butterfly = surface[:, i-1] - 2*surface[:, i] + surface[:, i+1]
        violations += (butterfly < -tolerance).sum()
    
    return {
        'butterfly_violations': violations,
        'violation_pct': violations / (surface.shape[0] * (len(strikes)-2)) * 100
    }

# ============= Feature Engineering Analysis =============

def analyze_feature_distributions(features_df):
    """
    compute distribution stats for each feature column
    input: features_df (pandas dataframe)
    output: dataframe of stats
    """
    results = {}
    
    for col in features_df.columns:
        data = features_df[col].dropna()
        results[col] = {
            'mean': data.mean(),
            'std': data.std(),
            'skew': data.skew(),
            'kurtosis': data.kurtosis(),
            'min': data.min(),
            'max': data.max(),
            'nulls': features_df[col].isnull().sum(),
            'unique': data.nunique()
        }
    
    return pd.DataFrame(results).T

def visualize_feature_vs_surface_stats(features_df, surface_stats_df):
    """
    plot and print feature/surface statistics and correlations
    input: features_df, surface_stats_df
    output: matplotlib plots, stats/correlation tables
    """
    # Part 1: Charts
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Feature correlation matrix
    corr_matrix = features_df.corr()
    sns.heatmap(corr_matrix, ax=axes[0, 0], cmap='coolwarm', center=0, 
                cbar_kws={'shrink': 0.6}, square=True,
                xticklabels=False, yticklabels=False)
    axes[0, 0].set_title('Feature Correlations (26x26)')
    
    # 2. Surface statistics over time
    if 'mean_iv' in surface_stats_df.columns:
        axes[0, 1].plot(surface_stats_df['mean_iv'], linewidth=0.5, label='Mean IV')
        axes[0, 1].plot(surface_stats_df['std_iv'], linewidth=0.5, label='Std IV')
        axes[0, 1].set_title('Surface Statistics Over Time')
        axes[0, 1].legend()
    
    # 3. Feature distributions (box plot)
    features_normalized = (features_df - features_df.mean()) / features_df.std()
    features_normalized.boxplot(ax=axes[0, 2], rot=90)
    axes[0, 2].set_title('Normalized Feature Distributions')
    axes[0, 2].tick_params(axis='x', labelsize=5)
    
    # 4. Top 10 most variable features (CV)
    feat_cv = (features_df.std() / features_df.mean().abs()).sort_values(ascending=False)[:10]
    axes[1, 0].bar(range(len(feat_cv)), feat_cv.values)
    axes[1, 0].set_xticks(range(len(feat_cv)))
    axes[1, 0].set_xticklabels(feat_cv.index, rotation=45, ha='right', fontsize=7)
    axes[1, 0].set_title('Top 10 Most Variable Features (CV)')
    axes[1, 0].set_ylabel('Coefficient of Variation')
    
    # 5. Skewness distribution
    feat_skew = features_df.skew()
    axes[1, 1].hist(feat_skew, bins=20, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--')
    axes[1, 1].set_title('Feature Skewness Distribution')
    axes[1, 1].set_xlabel('Skewness')
    axes[1, 1].set_ylabel('Count')
    
    # 6. Best correlation scatter
    if 'atm_iv' in surface_stats_df.columns:
        correlations = []
        for col in features_df.columns:
            corr = features_df[col].corr(surface_stats_df['atm_iv'])
            correlations.append((col, corr))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        best_feat = correlations[0][0]
        axes[1, 2].scatter(features_df[best_feat], surface_stats_df['atm_iv'], alpha=0.5, s=1)
        axes[1, 2].set_xlabel(best_feat, fontsize=8)
        axes[1, 2].set_ylabel('ATM IV')
        axes[1, 2].set_title(f'Best Correlation: r={correlations[0][1]:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    # Part 2: Tables and detailed statistics
    print("\n" + "="*80)
    print("FEATURE ANALYSIS TABLES")
    print("="*80)
    
    # Feature statistics table
    stats_df = pd.DataFrame({
        'Mean': features_df.mean(),
        'Std': features_df.std(),
        'Min': features_df.min(),
        'Max': features_df.max(),
        'Skew': features_df.skew(),
        'Kurt': features_df.kurtosis(),
        'CV': features_df.std() / features_df.mean().abs()
    })
    
    print("\n1. Feature Statistics Summary:")
    print(stats_df.round(4))
    
    # Feature-ATM IV correlations table
    if 'atm_iv' in surface_stats_df.columns:
        corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
        corr_df['Abs_Corr'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Corr', ascending=False)
        
        print("\n2. Feature-ATM IV Correlations (sorted by absolute value):")
        print(corr_df[['Feature', 'Correlation']].round(4))
        
        print("\n3. Top 5 Most Correlated Features:")
        print(corr_df.head()[['Feature', 'Correlation']].to_string(index=False))
        
        print("\n4. Features with |correlation| > 0.02:")
        significant = corr_df[corr_df['Abs_Corr'] > 0.02]
        print(f"Count: {len(significant)}")
        print(significant[['Feature', 'Correlation']].to_string(index=False))
    
    # Normalization recommendations
    print("\n5. Normalization Recommendations:")
    high_skew = stats_df[stats_df['Skew'].abs() > 2]
    print(f"Features with high skewness (|skew| > 2): {len(high_skew)}")
    if len(high_skew) > 0:
        print("Consider log transformation for:")
        print(high_skew.index.tolist())
    
    high_cv = stats_df[stats_df['CV'] > 1]
    print(f"\nFeatures with high CV (>1): {len(high_cv)}")
    if len(high_cv) > 0:
        print("Consider standardization for:")
        print(high_cv.index.tolist())
    
    return stats_df, corr_df if 'atm_iv' in surface_stats_df.columns else None   
def check_feature_stationarity(series, test='adf'):
    """
    test stationarity of feature time series using adf or kpss
    input: series (pandas series), test type
    output: dict with test results
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    if test == 'adf':
        result = adfuller(series.dropna())
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    elif test == 'kpss':
        result = kpss(series.dropna(), regression='c')
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] > 0.05,
            'critical_values': result[3]
        }
# Add to edav.py

def analyze_maturity_aggregation_impact_tensor(surface_tensor, taus, strikes):
    """
    analyze information loss from averaging across maturities in tensor
    input: surface_tensor, taus, strikes
    output: dict with per-maturity stats and info loss
    """
    """Analyze information loss from averaging across maturities using tensor data"""
    iv_surfaces = ops.convert_to_numpy(surface_tensor[:, :, :, -1])
    T, M, K = iv_surfaces.shape
    
    metrics_by_maturity = []
    for m in range(M):
        maturity_surface = iv_surfaces[:, m, :]
        
        metrics_by_maturity.append({
            'tau': taus[m],
            'skew': np.mean(maturity_surface[:, :K//2]) - np.mean(maturity_surface[:, K//2:]),
            'convexity': 2 * np.mean(maturity_surface[:, K//2]) - np.mean(maturity_surface[:, 0]) - np.mean(maturity_surface[:, -1]),
            'atm_level': np.mean(maturity_surface[:, K//2]),
            'mean_iv': np.mean(maturity_surface),
            'std_iv': np.std(maturity_surface)
        })
    
    df = pd.DataFrame(metrics_by_maturity)
    
    # Calculate information loss
    information_loss = {}
    for col in ['skew', 'convexity', 'atm_level', 'mean_iv']:
        std_across_maturities = df[col].std()
        mean_value = df[col].mean()
        loss_pct = (std_across_maturities / abs(mean_value) * 100) if mean_value != 0 else 0
        information_loss[col] = loss_pct
    
    return {
        'per_maturity_stats': df,
        'overall_mean': df.mean(),
        'information_loss_pct': information_loss,
        'max_loss_feature': max(information_loss, key=information_loss.get)
    }
def compute_feature_vif(features_df):
    """
    compute variance inflation factor for each feature
    input: features_df (pandas dataframe)
    output: dataframe of vif values
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_df.columns
    vif_data["VIF"] = [variance_inflation_factor(features_df.values, i) 
                       for i in range(len(features_df.columns))]
    return vif_data.sort_values('VIF', ascending=False)

#  Tensor Transformation Analysis 

def validate_pivot_transformation(original_df, pivot_df, value_col='market_iv'):
    """
    validate pivot transformation preserves values
    input: original_df, pivot_df, value_col
    output: dict with preservation stats
    """
    original_values = set(original_df[value_col].dropna())
    pivot_values = set(pivot_df.values.flatten())
    pivot_values.discard(np.nan)
    
    lost_values = original_values - pivot_values
    new_values = pivot_values - original_values
    
    return {
        'original_count': len(original_values),
        'pivot_count': len(pivot_values),
        'lost_values': len(lost_values),
        'new_values': len(new_values),
        'preservation_rate': len(pivot_values.intersection(original_values)) / len(original_values) * 100
    }

def analyze_surface_statistics(surface_tensor):
    """
    compute basic statistics for each surface in tensor
    input: surface_tensor (t, m, k, c)
    output: dataframe of stats
    """
    T, M, K = surface_tensor.shape[:3]
    stats_dict = {}
    
    for t in range(T):
        surface = surface_tensor[t, :, :, -1] if surface_tensor.ndim == 4 else surface_tensor[t]
        
        stats_dict[t] = {
            'mean': np.mean(surface),
            'std': np.std(surface),
            'skew': stats.skew(surface.flatten()),
            'kurtosis': stats.kurtosis(surface.flatten()),
            'atm_level': surface[M//2, K//2] if M > 0 and K > 0 else np.nan,
            'min': np.min(surface),
            'max': np.max(surface),
            'nan_count': np.isnan(surface).sum()
        }
    
    return pd.DataFrame(stats_dict).T

#  Train/Val/Test Split Analysis

def check_temporal_leakage(train_idx, val_idx, test_idx, lookback=20):
    """
    check for temporal leakage between train/val/test splits
    input: train_idx, val_idx, test_idx, lookback
    output: dict with leakage issues
    """
    leakage_issues = []
    
    if train_idx[-1] + lookback > val_idx[0]:
        leakage_issues.append(f"Train→Val leakage: {train_idx[-1] + lookback - val_idx[0]} overlapping points")
    
    if val_idx[-1] + lookback > test_idx[0]:
        leakage_issues.append(f"Val→Test leakage: {val_idx[-1] + lookback - test_idx[0]} overlapping points")
    
    return {
        'has_leakage': len(leakage_issues) > 0,
        'issues': leakage_issues,
        'safe_with_lookback': len(leakage_issues) == 0
    }

def analyze_split_distributions(train_data, val_data, test_data, channel=-1):
    """
    compare distributions of train/val/test splits using ks test
    input: train_data, val_data, test_data, channel
    output: dict with ks test results and summary stats
    """
    from scipy.stats import ks_2samp
    
    results = {}
    
    # Extract only the specified channel (default: -1 for IV)
    if train_data.ndim == 4:
        train_flat = train_data[:, :, :, channel].flatten()
        val_flat = val_data[:, :, :, channel].flatten()
        test_flat = test_data[:, :, :, channel].flatten()
    else:
        train_flat = train_data.flatten()
        val_flat = val_data.flatten()
        test_flat = test_data.flatten()
    
    results['train_vs_val'] = ks_2samp(train_flat, val_flat)
    results['train_vs_test'] = ks_2samp(train_flat, test_flat)
    results['val_vs_test'] = ks_2samp(val_flat, test_flat)
    
    results['summary'] = {
        'train_mean': np.mean(train_flat),
        'val_mean': np.mean(val_flat),
        'test_mean': np.mean(test_flat),
        'train_std': np.std(train_flat),
        'val_std': np.std(val_flat),
        'test_std': np.std(test_flat)
    }
    
    return results
#  Pointwise Transformation Analysis

def analyze_pointwise_distribution(pointwise_data, M, K):
    """
    analyze sample distribution for pointwise data
    input: pointwise_data, M, K
    output: dict with sample grid and balance stats
    """
    n_samples = len(pointwise_data)
    expected_surfaces = n_samples / (M * K)
    
    sample_grid = np.zeros((M, K))
    for i in range(n_samples):
        m_idx = i % M
        k_idx = (i // M) % K
        sample_grid[m_idx, k_idx] += 1
    
    return {
        'total_samples': n_samples,
        'expected_surfaces': expected_surfaces,
        'samples_per_point': sample_grid,
        'min_samples': sample_grid.min(),
        'max_samples': sample_grid.max(),
        'balance_ratio': sample_grid.min() / sample_grid.max()
    }

def validate_feature_target_alignment(features, targets, surface_idx, point_idx):
    """
    check alignment between features and targets for given indices
    input: features, targets, surface_idx, point_idx
    output: dict with consistency check
    """
    feature_value = features[surface_idx * 110 + point_idx]
    target_value = targets[surface_idx * 110 + point_idx]
    
    expected_feature = features[surface_idx * 110:(surface_idx + 1) * 110]
    
    return {
        'feature_consistent': np.all(expected_feature == feature_value),
        'target_value': target_value,
        'feature_value': feature_value
    }

#  Reconstruction Analysis

def analyze_reconstruction_errors(original, reconstructed, M, K):
    """
    compute reconstruction error metrics for surfaces
    input: original, reconstructed, M, K
    output: dict with error metrics and heatmap
    """
    errors = np.abs(original - reconstructed)
    mse = np.mean(errors**2)
    
    per_maturity = errors.mean(axis=2) if errors.ndim == 3 else errors.mean(axis=1)
    per_strike = errors.mean(axis=1) if errors.ndim == 3 else errors.mean(axis=0)
    
    error_heatmap = errors.mean(axis=0) if errors.ndim == 3 else errors
    
    return {
        'global_mse': mse,
        'global_rmse': np.sqrt(mse),
        'global_mae': np.mean(errors),
        'per_maturity_error': per_maturity,
        'per_strike_error': per_strike,
        'error_heatmap': error_heatmap,
        'worst_points': np.unravel_index(errors.argmax(), errors.shape),
        'best_points': np.unravel_index(errors.argmin(), errors.shape)
    }

def identify_problem_regions(errors, threshold_pct=90):
    """
    identify regions with high reconstruction error
    input: errors (array), threshold_pct
    output: dict with mask and stats
    """
    threshold = np.percentile(errors.flatten(), threshold_pct)
    problem_mask = errors > threshold
    
    return {
        'problem_mask': problem_mask,
        'problem_count': problem_mask.sum(),
        'problem_pct': problem_mask.mean() * 100,
        'threshold_value': threshold
    }

# ----------------------- Comparative Analysis

def compute_information_metrics(data):
    """
    compute entropy, variance, unique values, dynamic range for data
    input: data (array)
    output: dict with information metrics
    """
    from scipy.stats import entropy
    
    flat_data = data.flatten()
    hist, bins = np.histogram(flat_data[~np.isnan(flat_data)], bins=50)
    hist = hist / hist.sum()
    
    return {
        'entropy': entropy(hist),
        'variance': np.var(flat_data[~np.isnan(flat_data)]),
        'unique_values': len(np.unique(flat_data[~np.isnan(flat_data)])),
        'dynamic_range': np.ptp(flat_data[~np.isnan(flat_data)])
    }

def compare_pipeline_stages(stage_data_dict):
    """
    compare information metrics across pipeline stages
    input: stage_data_dict (dict of arrays)
    output: dataframe of metrics and changes
    """
    comparison = {}
    
    for stage_name, data in stage_data_dict.items():
        comparison[stage_name] = compute_information_metrics(data)
    
    df = pd.DataFrame(comparison).T
    df['entropy_change'] = df['entropy'].pct_change() * 100
    df['variance_change'] = df['variance'].pct_change() * 100
    
    return df

# ----------------------- Visualization 

def plot_surface_heatmap(surface, title="Volatility Surface", maturities=None, strikes=None):
    """
    plot heatmap for volatility surface
    input: surface (2d array), title, maturities, strikes
    output: matplotlib plot
    """
    plt.figure(figsize=(10, 6))
    
    if maturities is not None and strikes is not None:
        sns.heatmap(surface, xticklabels=strikes, yticklabels=maturities, 
                   cmap='viridis', annot=False, fmt='.3f')
    else:
        sns.heatmap(surface, cmap='viridis', annot=False, fmt='.3f')
    
    plt.title(title)
    plt.xlabel("Strike")
    plt.ylabel("Maturity")
    plt.tight_layout()
    plt.show()

def plot_error_analysis(errors_dict):
    """
    plot error metrics for surface reconstruction
    input: errors_dict (dict)
    output: matplotlib plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(errors_dict['per_maturity_error'])
    axes[0, 0].set_title("Error by Maturity")
    axes[0, 0].set_xlabel("Maturity Index")
    axes[0, 0].set_ylabel("Mean Error")
    
    axes[0, 1].plot(errors_dict['per_strike_error'])
    axes[0, 1].set_title("Error by Strike")
    axes[0, 1].set_xlabel("Strike Index")
    axes[0, 1].set_ylabel("Mean Error")
    
    sns.heatmap(errors_dict['error_heatmap'], cmap='Reds', ax=axes[1, 0])
    axes[1, 0].set_title("Error Heatmap")
    
    axes[1, 1].hist(errors_dict['error_heatmap'].flatten(), bins=50, edgecolor='black')
    axes[1, 1].set_title("Error Distribution")
    axes[1, 1].set_xlabel("Error")
    axes[1, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def create_diagnostic_dashboard(data_dict):
    """
    create dashboard of multiple plots for diagnostic analysis
    input: data_dict (dict of arrays/dataframes)
    output: matplotlib figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))
    
    plot_idx = 0
    for key, data in list(data_dict.items())[:9]:
        if isinstance(data, pd.DataFrame):
            data.plot(ax=axes[plot_idx], legend=False)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            im = axes[plot_idx].imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=axes[plot_idx])
        elif isinstance(data, (list, np.ndarray)):
            axes[plot_idx].plot(data)
        else:
            axes[plot_idx].text(0.5, 0.5, str(data), ha='center', va='center')
        
        axes[plot_idx].set_title(key)
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()

# -------------- generate reports 

def generate_edav_summary(analysis_results):
    """
    generate text summary of edav analysis results
    input: analysis_results (dict)
    output: string summary
    """
    summary = []
    summary.append("="*50)
    summary.append("EDAV PIPELINE ANALYSIS SUMMARY")
    summary.append("="*50)
    
    for section, results in analysis_results.items():
        summary.append(f"\n{section}:")
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    summary.append(f"  {key}: {value:.4f}")
                else:
                    summary.append(f"  {key}: {value}")
        else:
            summary.append(f"  {results}")
    
    return "\n".join(summary)

def export_analysis_results(analysis_dict, output_dir="edav_results"):
    """
    export analysis results to csv, npy, or json files
    input: analysis_dict, output_dir
    output: path to results directory
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    for key, value in analysis_dict.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(f"{output_dir}/{key}.csv")
        elif isinstance(value, np.ndarray):
            np.save(f"{output_dir}/{key}.npy", value)
        elif isinstance(value, dict):
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(value, f, indent=2, default=str)
    
    return f"Results exported to {output_dir}/"

# -------------- Raw Data Analysis --------------

def analyze_raw_data_quality(df_raw):
    """
    analyze missing values, date coverage, and iv stats in raw dataframe
    input: df_raw (pandas dataframe)
    output: dict with quality metrics
    """
    missing_by_col = df_raw.isnull().sum()
    missing_pct = (missing_by_col / len(df_raw) * 100).round(2)
    
    date_stats = {
        'unique_dates': df_raw['date'].nunique(),
        'date_range': f"{df_raw['date'].min()} to {df_raw['date'].max()}",
        'expected_business_days': pd.bdate_range(df_raw['date'].min(), df_raw['date'].max()).nunique(),
        'coverage': df_raw['date'].nunique() / pd.bdate_range(df_raw['date'].min(), df_raw['date'].max()).nunique()
    }
    
    expected_points = df_raw['maturity'].nunique() * df_raw['rel_strike'].nunique()
    actual_by_date = df_raw.groupby('date').size()
    
    iv_stats = df_raw['market_iv'].describe()
    
    return {
        'missing_values': missing_pct[missing_pct > 0],
        'date_stats': date_stats,
        'points_per_date': actual_by_date.describe(),
        'expected_points': expected_points,
        'iv_stats': iv_stats,
        'completeness_pct': (actual_by_date / expected_points * 100).describe()
    }
    
def visualize_missing_patterns(df_raw):
    """
    plot missing data patterns and completeness in raw dataframe
    input: df_raw (pandas dataframe)
    output: matplotlib plots and completeness stats
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    missing_by_col = df_raw.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if len(missing_by_col) > 0:
        axes[0, 0].barh(range(len(missing_by_col)), missing_by_col.values)
        axes[0, 0].set_yticks(range(len(missing_by_col)))
        axes[0, 0].set_yticklabels(missing_by_col.index)
        axes[0, 0].set_xlabel('Missing Count')
        axes[0, 0].set_title('Missing Values by Column')
        for i, v in enumerate(missing_by_col.values):
            axes[0, 0].text(v, i, f' {v:,.0f} ({v/len(df_raw)*100:.1f}%)', va='center')
    else:
        axes[0, 0].text(0.5, 0.5, 'No missing values', ha='center', va='center')
        axes[0, 0].set_title('Missing Values by Column')
    
    completeness_by_date = df_raw.groupby('date').apply(
        lambda x: len(x) / (df_raw['maturity'].nunique() * df_raw['rel_strike'].nunique()) * 100
    )
    axes[0, 1].plot(pd.to_datetime(completeness_by_date.index), completeness_by_date.values, linewidth=0.5)
    axes[0, 1].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100% Complete')
    axes[0, 1].axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Completeness %')
    axes[0, 1].set_title('Surface Completeness Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    pivot_missing = df_raw.pivot_table(
        index='maturity', 
        columns='rel_strike', 
        values='market_iv', 
        aggfunc=lambda x: x.isnull().mean() * 100
    )
    
    im = axes[1, 0].imshow(pivot_missing, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
    axes[1, 0].set_xticks(range(len(pivot_missing.columns)))
    axes[1, 0].set_xticklabels([f'{x:.2f}' for x in pivot_missing.columns], rotation=45)
    axes[1, 0].set_yticks(range(len(pivot_missing.index)))
    axes[1, 0].set_yticklabels(pivot_missing.index)
    axes[1, 0].set_xlabel('Strike')
    axes[1, 0].set_ylabel('Maturity')
    axes[1, 0].set_title('Missing IV % by Maturity-Strike')
    plt.colorbar(im, ax=axes[1, 0])
    
    dates = pd.Series(pd.to_datetime(df_raw['date'].unique())).sort_values()
    gaps = dates.diff().dt.days.dropna()
    
    axes[1, 1].hist(gaps, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=1, color='g', linestyle='--', label='1 day')
    axes[1, 1].axvline(x=3, color='orange', linestyle='--', label='3 days')
    axes[1, 1].set_xlabel('Gap (days)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Temporal Gaps')
    axes[1, 1].legend()
    
    axes[1, 1].text(0.98, 0.98, 
                    f'Mean: {gaps.mean():.1f} days\nMax: {gaps.max():.0f} days',
                    transform=axes[1, 1].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Missing Data Patterns Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {
        'completeness_by_date': completeness_by_date,
        'missing_by_maturity_strike': pivot_missing,
        'temporal_gaps': gaps
    }

def detect_iv_outliers_comprehensive(df_raw, iv_col='market_iv'):
    """
    detect iv outliers using zscore, iqr, and isolation forest
    input: df_raw (pandas dataframe), iv_col
    output: dataframe with outlier flags
    """
    iv_clean = df_raw[iv_col].dropna()
    
    z_scores = np.abs(stats.zscore(iv_clean))
    z_outliers = z_scores > 3
    
    Q1 = iv_clean.quantile(0.25)
    Q3 = iv_clean.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = (iv_clean < (Q1 - 1.5 * IQR)) | (iv_clean > (Q3 + 1.5 * IQR))
    
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_outliers = iso_forest.fit_predict(iv_clean.values.reshape(-1, 1)) == -1
    
    outlier_summary = pd.DataFrame({
        'iv_value': iv_clean,
        'z_score': z_scores,
        'z_outlier': z_outliers,
        'iqr_outlier': iqr_outliers,
        'iso_outlier': iso_outliers
    })
    
    outlier_summary['any_outlier'] = outlier_summary[['z_outlier', 'iqr_outlier', 'iso_outlier']].any(axis=1)
    
    return outlier_summary


def check_no_arbitrage_iv_monotonicity(surface: np.ndarray, taus: np.ndarray, strikes: np.ndarray):
    """
    check smile and calendar arbitrage using finite differences
    input: surface (m, k), taus, strikes
    output: smile and calendar violation counts
    """
    
    
    smile_violations = 0
    calendar_violations = 0
    M, K = surface.shape

    # Smile convexity: ∂²IV/∂m² ≥ 0
    for i in range(M):
        for j in range(1, K - 1):
            fwd_diff = surface[i, j+1] - surface[i, j]
            back_diff = surface[i, j] - surface[i, j-1]
            if (fwd_diff - back_diff) < 0:
                smile_violations += 1

    # Calendar: IV monotonicity
    for j in range(K):
        for i in range(1, M):
            if surface[i, j] < surface[i-1, j]:
                calendar_violations += 1

    return smile_violations, calendar_violations

def check_no_arbitrage_total_variance(surface: np.ndarray, taus: np.ndarray, strikes: np.ndarray, tol=1e-6):
    """
    check total variance monotonicity and butterfly spread conditions
    input: surface (m, k), taus, strikes, tol
    output: smile and calendar violation counts
    """
   
    smile_violations = 0
    calendar_violations = 0
    M, K = surface.shape
    
    # Butterfly spreads
    for i in range(M):
        for j in range(1, K-1):
            butterfly = surface[i, j+1] - 2*surface[i, j] + surface[i, j-1]
            if butterfly < -tol:
                smile_violations += 1
    
    # Total variance monotonicity
    for j in range(K):
        for i in range(1, M):
            total_var_curr = surface[i, j]**2 * taus[i]
            total_var_prev = surface[i-1, j]**2 * taus[i-1]
            if total_var_curr < total_var_prev - tol:
                calendar_violations += 1
    
    return smile_violations, calendar_violations