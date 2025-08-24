import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def analyze_decoder_comprehensive(Y_true, Y_pred, taus, rel_strikes, date_idx=-1, 
                                 dates=None, title="Decoder Analysis"):
    """
    comprehensive decoder analysis with visualization and tabular summary
    input: Y_true, Y_pred (arrays), taus, rel_strikes, date_idx, dates, title
    output: summary dataframe, plots
    """
    
    # Ensure proper shapes
    if Y_true.ndim == 2:
        M, K = len(taus), len(rel_strikes)
        Y_true = Y_true.reshape(-1, M, K)
    
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)
    
    # Handle negative indexing
    if date_idx < 0:
        date_idx = len(Y_true) + date_idx
    
    N, M, K = Y_true.shape
    
    # Extract single date surfaces for detailed analysis
    true_surface = Y_true[date_idx]  # (M, K)
    pred_surface = Y_pred[date_idx]  # (M, K)
    
    # Calculate all error metrics
    errors = Y_pred - Y_true
    abs_errors = np.abs(errors)
    
    overall_rmse = np.sqrt(mean_squared_error(Y_true.reshape(N, -1), Y_pred.reshape(N, -1)))
    overall_mae = mean_absolute_error(Y_true.reshape(N, -1), Y_pred.reshape(N, -1))
    
    # ATM analysis (corrected)
    atm_idx = np.argmin(np.abs(rel_strikes - 1.0))
    true_atm = true_surface[:, atm_idx]
    pred_atm = pred_surface[:, atm_idx]
    atm_rmse = np.sqrt(mean_squared_error(true_atm, pred_atm))
    atm_mae = np.mean(abs_errors[:, :, atm_idx])
    
    # Error by maturity and strike
    rmse_by_maturity = np.sqrt(np.mean((Y_true - Y_pred)**2, axis=(0, 2)))  # Average over N, K
    rmse_by_strike = np.sqrt(np.mean((Y_true - Y_pred)**2, axis=(0, 1)))    # Average over N, M
    
    # Short vs long term analysis
    short_mask = np.array(taus) < 1.0
    if np.any(short_mask) and np.any(~short_mask):
        short_rmse = np.sqrt(mean_squared_error(Y_true[:, short_mask, :].flatten(), 
                                               Y_pred[:, short_mask, :].flatten()))
        long_rmse = np.sqrt(mean_squared_error(Y_true[:, ~short_mask, :].flatten(), 
                                              Y_pred[:, ~short_mask, :].flatten()))
        ratio = short_rmse / long_rmse
        short_term_mae = np.mean(abs_errors[:, short_mask, :])
        long_term_mae = np.mean(abs_errors[:, ~short_mask, :])
    else:
        short_rmse = long_rmse = ratio = np.nan
        short_term_mae = long_term_mae = np.nan
    
    # ITM/OTM analysis
    itm_mask = rel_strikes < 1.0
    otm_mask = rel_strikes > 1.0
    itm_rmse = np.sqrt(np.mean((Y_true[:, :, itm_mask] - Y_pred[:, :, itm_mask])**2))
    otm_rmse = np.sqrt(np.mean((Y_true[:, :, otm_mask] - Y_pred[:, :, otm_mask])**2))
    
    abs_errors_flat = abs_errors.flatten()
    
    # === CREATE COMPREHENSIVE VISUALIZATION ===
    
    # Create main analysis figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4, 
                         height_ratios=[1, 1, 1, 0.3], width_ratios=[1, 1, 1, 1])
    
    # Title with date
    date_str = f"Date {date_idx}" if dates is None else str(dates[date_idx])
    fig.suptitle(f"{title} - {date_str}", fontsize=16, fontweight='bold')
    
    # === 1. ATM Term Structure ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(taus, true_atm, 'o-', color='blue', linewidth=2, markersize=5, label='True')
    ax1.plot(taus, pred_atm, 's-', color='orange', linewidth=2, markersize=4, label='Pred')
    ax1.fill_between(taus, true_atm, pred_atm, alpha=0.3, color='gray')
    ax1.set_title('ATM Term Structure', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time to Maturity')
    ax1.set_ylabel('Implied Volatility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 2. Error Heatmap ===
    ax2 = fig.add_subplot(gs[0, 1])
    errors_surface = np.abs(true_surface - pred_surface)
    im = ax2.imshow(errors_surface, cmap='Reds', aspect='auto', origin='lower')
    ax2.set_title('Absolute Error Heatmap', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Strike Index')
    ax2.set_ylabel('Maturity Index')
    ax2.set_xticks(range(0, K, 2))
    ax2.set_xticklabels([f'{rel_strikes[i]:.1f}' for i in range(0, K, 2)])
    ax2.set_yticks(range(0, M, 2))
    ax2.set_yticklabels([f'{taus[i]:.1f}' for i in range(0, M, 2)])
    plt.colorbar(im, ax=ax2, label='Absolute Error', shrink=0.8)
    
    # === 3. 3D Surface Comparison ===
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    M_grid, K_grid = np.meshgrid(range(K), range(M))
    ax3.plot_wireframe(M_grid, K_grid, true_surface, color='blue', alpha=0.6, linewidth=1)
    ax3.plot_wireframe(M_grid, K_grid, pred_surface, color='orange', alpha=0.6, linewidth=1)
    ax3.set_title('3D Surface Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Strike Index')
    ax3.set_ylabel('Maturity Index')
    ax3.set_zlabel('IV')
    
    # === 4. Error Distribution ===
    ax4 = fig.add_subplot(gs[0, 3])
    n, bins, patches = ax4.hist(abs_errors_flat, bins=25, alpha=0.7, color='red', edgecolor='black')
    mean_error = np.mean(abs_errors_flat)
    median_error = np.median(abs_errors_flat)
    p95_error = np.percentile(abs_errors_flat, 95)
    ax4.axvline(mean_error, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    ax4.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.4f}')
    ax4.axvline(p95_error, color='red', linestyle='--', linewidth=2, label=f'95th: {p95_error:.4f}')
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # === 5. Error by Maturity ===
    ax5 = fig.add_subplot(gs[1, 0])
    error_by_maturity = np.mean(abs_errors, axis=(0, 2))
    std_by_maturity = np.std(abs_errors, axis=(0, 2))
    ax5.errorbar(taus, error_by_maturity, yerr=std_by_maturity, 
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)
    ax5.set_title('Mean Error by Maturity', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time to Maturity')
    ax5.set_ylabel('Mean Absolute Error')
    ax5.grid(True, alpha=0.3)
    
    # === 6. Error by Strike ===
    ax6 = fig.add_subplot(gs[1, 1])
    error_by_strike = np.mean(abs_errors, axis=(0, 1))
    std_by_strike = np.std(abs_errors, axis=(0, 1))
    ax6.errorbar(rel_strikes, error_by_strike, yerr=std_by_strike,
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)
    ax6.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
    ax6.set_title('Mean Error by Strike', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Relative Strike')
    ax6.set_ylabel('Mean Absolute Error')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # === 7. Error Contour Map ===
    ax7 = fig.add_subplot(gs[1, 2])
    M_cont, K_cont = np.meshgrid(rel_strikes, taus)
    contour = ax7.contourf(M_cont, K_cont, errors_surface, levels=15, cmap='Reds')
    ax7.contour(M_cont, K_cont, errors_surface, levels=15, colors='black', alpha=0.4, linewidths=0.5)
    ax7.set_title('Error Contour Map', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Relative Strike')
    ax7.set_ylabel('Time to Maturity')
    plt.colorbar(contour, ax=ax7, label='Error', shrink=0.8)
    
    # === 8. Residuals vs Fitted ===
    ax8 = fig.add_subplot(gs[1, 3])
    pred_flat = pred_surface.flatten()
    residuals = (true_surface - pred_surface).flatten()
    ax8.scatter(pred_flat, residuals, alpha=0.6, s=15)
    ax8.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax8.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Predicted IV')
    ax8.set_ylabel('Residuals')
    ax8.grid(True, alpha=0.3)
    
    # === 9. RMSE by Maturity Bar Chart ===
    ax9 = fig.add_subplot(gs[2, 0])
    bars = ax9.bar(range(M), rmse_by_maturity, color='skyblue', edgecolor='black')
    ax9.set_title('RMSE by Maturity', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Maturity Index')
    ax9.set_ylabel('RMSE')
    ax9.set_xticks(range(M))
    ax9.set_xticklabels([f'{tau:.2f}' for tau in taus], rotation=45)
    ax9.grid(True, alpha=0.3)
    
    # === 10. RMSE by Strike Bar Chart ===
    ax10 = fig.add_subplot(gs[2, 1])
    bars = ax10.bar(range(K), rmse_by_strike, color='lightcoral', edgecolor='black')
    ax10.set_title('RMSE by Strike', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Strike Index')
    ax10.set_ylabel('RMSE')
    ax10.set_xticks(range(K))
    ax10.set_xticklabels([f'{strike:.1f}' for strike in rel_strikes], rotation=45)
    ax10.axvline(x=atm_idx, color='red', linestyle='--', alpha=0.7, label='ATM')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # === 11. Performance Comparison ===
    ax11 = fig.add_subplot(gs[2, 2])
    categories = ['Overall', 'ATM', 'ITM', 'OTM', 'Short-term', 'Long-term']
    values = [overall_rmse, atm_rmse, itm_rmse, otm_rmse, short_rmse, long_rmse]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    bars = ax11.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax11.set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    ax11.set_ylabel('RMSE')
    ax11.tick_params(axis='x', rotation=45)
    ax11.grid(True, alpha=0.3)
    
    # === 12. Time Series Analysis ===
    ax12 = fig.add_subplot(gs[2, 3])
    short_atm_errors = abs_errors[:, taus < 0.5, atm_idx].mean(axis=1)
    ax12.plot(short_atm_errors, linewidth=1, alpha=0.7)
    ax12.set_xlabel('Date Index')
    ax12.set_ylabel('Short-term ATM Error')
    ax12.set_title('Short-term ATM Error Series')
    ax12.grid(True, alpha=0.3)
    if date_idx is not None:
        ax12.axvline(x=date_idx, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    # === CREATE AND DISPLAY SUMMARY DATAFRAMES ===
    
    # Calculate additional metrics for summary
    worst_maturity_idx = np.argmax(rmse_by_maturity)
    worst_strike_idx = np.argmax(rmse_by_strike)
    
    # Create comprehensive summary
    summary_data = {
        'Overall Performance': {
            'RMSE': f"{overall_rmse:.6f}",
            'MAE': f"{overall_mae:.6f}",
            'Max Error': f"{np.max(abs_errors_flat):.6f}",
            'Mean Error': f"{np.mean(abs_errors_flat):.6f}",
            'Median Error': f"{np.median(abs_errors_flat):.6f}",
            '95th Percentile': f"{np.percentile(abs_errors_flat, 95):.6f}"
        },
        'Category Performance': {
            'ATM RMSE': f"{atm_rmse:.6f}",
            'ITM RMSE': f"{itm_rmse:.6f}",
            'OTM RMSE': f"{otm_rmse:.6f}",
            'Short-term RMSE': f"{short_rmse:.6f}",
            'Long-term RMSE': f"{long_rmse:.6f}",
            'Short/Long Ratio': f"{ratio:.3f}"
        },
        'Best/Worst Performance': {
            'Best Maturity': f"τ={taus[np.argmin(rmse_by_maturity)]:.2f} (RMSE: {np.min(rmse_by_maturity):.6f})",
            'Worst Maturity': f"τ={taus[worst_maturity_idx]:.2f} (RMSE: {np.max(rmse_by_maturity):.6f})",
            'Best Strike': f"K={rel_strikes[np.argmin(rmse_by_strike)]:.1f} (RMSE: {np.min(rmse_by_strike):.6f})",
            'Worst Strike': f"K={rel_strikes[worst_strike_idx]:.1f} (RMSE: {np.max(rmse_by_strike):.6f})",
            'ATM Performance': f"K=1.0 (RMSE: {rmse_by_strike[atm_idx]:.6f})",
            'Overall Rank': f"ATM ranks #{np.argsort(rmse_by_strike)[atm_idx]+1} of {K}"
        },
        'Diagnostic Flags': {
            'Short-term Bias': f"{'YES' if ratio > 1.2 else 'NO'} (ratio: {ratio:.3f})",
            'ATM Issues': f"{'YES' if atm_rmse > 1.2 * overall_rmse else 'NO'} (ATM/Overall: {atm_rmse/overall_rmse:.2f}x)",
            'Smile Asymmetry': f"{'YES' if abs(itm_rmse - otm_rmse) > 0.01 else 'NO'} (|ITM-OTM|: {abs(itm_rmse - otm_rmse):.6f})",
            'High Error Variance': f"{'YES' if np.std(abs_errors_flat) > 0.005 else 'NO'} (std: {np.std(abs_errors_flat):.6f})",
            'Model Quality': 'EXCELLENT' if overall_rmse < 0.01 else 'GOOD' if overall_rmse < 0.02 else 'FAIR',
            'Recommendation': 'Production Ready' if overall_rmse < 0.015 and ratio < 1.5 else 'Needs Improvement'
        }
    }
    
    summary_df = pd.DataFrame(summary_data)
    
   
    

    display(summary_df)
    
  
    return summary_df

def summarize_decoder_errors(Y_true, Y_pred, taus, rel_strikes, name="Decoder", model=None):
    """
    extract key metrics for decoder experiment tracking (no charts)
    input: Y_true, Y_pred (arrays), taus, rel_strikes, name, model
    output: dict of summary metrics
    """
    
    # Ensure proper shapes
    if Y_true.ndim == 2:
        M, K = len(taus), len(rel_strikes)
        Y_true = Y_true.reshape(-1, M, K)
    
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)
    N, M, K = Y_true.shape
    
    # Calculate key metrics only (no visualization)
    overall_rmse = np.sqrt(mean_squared_error(Y_true.reshape(N, -1), Y_pred.reshape(N, -1)))
    overall_mae = mean_absolute_error(Y_true.reshape(N, -1), Y_pred.reshape(N, -1))
    
    # ATM analysis
    atm_idx = np.argmin(np.abs(rel_strikes - 1.0))
    atm_rmse = np.sqrt(mean_squared_error(Y_true[:, :, atm_idx].flatten(), Y_pred[:, :, atm_idx].flatten()))
    atm_mae = np.mean(np.abs(Y_true[:, :, atm_idx] - Y_pred[:, :, atm_idx]))
    
    # ITM/OTM analysis
    itm_mask = rel_strikes < 1.0
    otm_mask = rel_strikes > 1.0
    itm_rmse = np.sqrt(np.mean((Y_true[:, :, itm_mask] - Y_pred[:, :, itm_mask])**2))
    otm_rmse = np.sqrt(np.mean((Y_true[:, :, otm_mask] - Y_pred[:, :, otm_mask])**2))
    
    # Short vs long term
    short_mask = np.array(taus) < 1.0
    if np.any(short_mask) and np.any(~short_mask):
        short_rmse = np.sqrt(mean_squared_error(Y_true[:, short_mask, :].flatten(), 
                                               Y_pred[:, short_mask, :].flatten()))
        long_rmse = np.sqrt(mean_squared_error(Y_true[:, ~short_mask, :].flatten(), 
                                              Y_pred[:, ~short_mask, :].flatten()))
        ratio = short_rmse / long_rmse
    else:
        ratio = np.nan
    
    abs_errors = np.abs(Y_true - Y_pred)
    
    # Extract key metrics for comparison
    summary = {
        'Name': name,
        'Overall_RMSE': overall_rmse,
        'Overall_MAE': overall_mae,
        'ATM_RMSE': atm_rmse,
        'ATM_MAE': atm_mae,
        'ITM_RMSE': itm_rmse,
        'OTM_RMSE': otm_rmse,
        'Short_Long_Ratio': ratio,
        'Max_Error': np.max(abs_errors),
        'Short_Term_Bias': ratio > 1.2 if not np.isnan(ratio) else False,
        'ATM_Issues': atm_mae > 1.2 * overall_mae,
        'Model_Quality': 'EXCELLENT' if overall_rmse < 0.01 else 'GOOD' if overall_rmse < 0.02 else 'FAIR',
        'Production_Ready': overall_rmse < 0.015 and (ratio < 1.5 if not np.isnan(ratio) else True)
    }
    
    return summary

def collect_decoder_summaries(*summaries):
    """
    compare multiple decoder experiments and return summary table
    input: *summaries (dicts)
    output: pandas dataframe comparison
    """
    
    if len(summaries) == 0:
        return pd.DataFrame()
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(summaries)
    
    # Sort by overall RMSE (best first)
    comparison_df = comparison_df.sort_values('Overall_RMSE').reset_index(drop=True)
    
    # Add ranking
    comparison_df.insert(1, 'Rank', range(1, len(comparison_df) + 1))
    
    # Format numerical columns
    numeric_cols = ['Overall_RMSE', 'Overall_MAE', 'ATM_RMSE', 'ATM_MAE', 
                   'ITM_RMSE', 'OTM_RMSE', 'Short_Long_Ratio', 'Max_Error']
    
    for col in numeric_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "N/A")
    
    return comparison_df