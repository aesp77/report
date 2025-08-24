# utils/moe_weights.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_expert_bias_matrix(calendar_multiplier=1.5, strike_bands=[0.9, 1.1], 
                                taus=None, strikes=None, progress=0.0, num_experts=7):
    
    if taus is None:
        taus = np.array([0.083, 0.167, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    if strikes is None:
        strikes = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    
    M, K = len(taus), len(strikes)
    
    # annealing
    max_bias_start, max_bias_end = 12.0, 4.0
    moderate_bias_start, moderate_bias_end = 4.0, 1.0
    
    max_bias = max_bias_end + (max_bias_start - max_bias_end) * (1.0 - progress)
    moderate_bias = moderate_bias_end + (moderate_bias_start - moderate_bias_end) * (1.0 - progress)
    
    print(f"bias levels: max={max_bias:.1f}, moderate={moderate_bias:.1f}")
    
    expert_biases = {}
    
    # 4 maturity experts (0-3)
    tau_bands = [0.3, 0.75, 2, 3.5]
    bias_strengths = [
        max_bias,              # ultra-short
        moderate_bias,         # short-medium
        moderate_bias + 2.0,   # problem zone
        moderate_bias + 1.0,   # long-term
        moderate_bias + 1.5    # background
    ]
    
    for expert_id in range(4):
        bias_matrix = np.zeros((M, K))
        
        for i, tau in enumerate(taus):
            if expert_id == 0:  # ultra-short
                if tau < tau_bands[0]:
                    bias_matrix[i, :] = bias_strengths[0]
            elif expert_id == 1:  # short-medium
                if tau_bands[0] <= tau < tau_bands[1]:
                    bias_matrix[i, :] = bias_strengths[1]
            elif expert_id == 2:  # problem zone
                if tau_bands[1] <= tau < tau_bands[2]:
                    bias_matrix[i, :] = bias_strengths[2]
            elif expert_id == 3:  # long-term
                if tau >= tau_bands[2]:
                    bias_matrix[i, :] = bias_strengths[3]
        
        expert_biases[f'expert_{expert_id}_mat'] = bias_matrix
    
    # 3 free experts (4-6) for strike specialization
    for expert_id in range(4, 7):
        bias_matrix = np.zeros((M, K))
        
        for i, tau in enumerate(taus):
            for j, strike in enumerate(strikes):
                
                if expert_id == 4:  # itm
                    if strike < strike_bands[0]:
                        bias_matrix[i, j] = max_bias * (1 - progress)
                        
                elif expert_id == 5:  # atm with calendar
                    if strike_bands[0] <= strike <= strike_bands[1]:
                        calendar_strength = max_bias * 1.5 * (1.0 - progress)
                        
                        if tau < 0.3:
                            boost = calendar_strength
                        elif 0.3 <= tau < 0.85:
                            boost = moderate_bias
                        elif 0.85 <= tau < 2:
                            boost = 2.0
                        elif tau >= 2:
                            boost = moderate_bias
                        else:
                            boost = 0
                        
                        bias_matrix[i, j] = 1.0 + boost
                        
                elif expert_id == 6:  # otm
                    if strike > strike_bands[1]:
                        bias_matrix[i, j] = max_bias * 2.5 * (1 - progress)
        
        expert_biases[f'expert_{expert_id}_strike'] = bias_matrix
    
    return expert_biases, taus, strikes

def plot_expert_bias_matrices(calendar_multiplier=1.5, strike_bands=[0.9, 1.1], 
                              progress=0.0, figsize=(21, 10)):
    
    expert_biases, taus, strikes = calculate_expert_bias_matrix(
        calendar_multiplier, strike_bands, progress=progress)
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(f'7 expert bias matrices (progress: {progress:.1f})', fontsize=14)
    
    cmap = 'YlOrRd'
    
    expert_names = [
        'e0: ultra-short', 'e1: short-med', 'e2: problem', 'e3: long',
        'e4: itm', 'e5: atm', 'e6: otm'
    ]
    
    for idx, (expert_key, bias_matrix) in enumerate(expert_biases.items()):
        if idx < 7:
            row, col = idx // 4, idx % 4
            ax = axes[row, col]
            
            im = ax.imshow(bias_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
            
            ax.set_xticks(range(0, len(strikes), 2))
            ax.set_xticklabels([f'{s:.1f}' for s in strikes[::2]], rotation=45)
            ax.set_yticks(range(0, len(taus), 2))
            ax.set_yticklabels([f'{t:.2f}' for t in taus[::2]])
            
            ax.set_xlabel('strike')
            ax.set_ylabel('maturity')
            ax.set_title(expert_names[idx], fontsize=10)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # show max bias
            max_val = bias_matrix.max()
            if max_val > 0:
                for i in range(0, len(taus), 3):
                    for j in range(0, len(strikes), 3):
                        if bias_matrix[i, j] > 0.1:
                            color = 'white' if bias_matrix[i, j] > max_val * 0.6 else 'black'
                            ax.text(j, i, f'{bias_matrix[i, j]:.0f}', 
                                   ha='center', va='center', color=color, fontsize=7)
    
    # remove empty subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # summary
    print("\nexpert bias summary:")
    for idx, (expert_key, bias_matrix) in enumerate(expert_biases.items()):
        if idx < 7:
            active = np.sum(bias_matrix > 0)
            total = bias_matrix.size
            max_bias = bias_matrix.max()
            avg_bias = bias_matrix[bias_matrix > 0].mean() if active > 0 else 0
            
            print(f"{expert_names[idx]:<15} | active: {active:3d}/{total:3d} ({active/total*100:4.1f}%) | "
                  f"max: {max_bias:5.1f} | avg: {avg_bias:5.1f}")

# test visualization
if __name__ == "__main__":
    plot_expert_bias_matrices(progress=0.0)
    print("\nwith progress=0.5:")
    plot_expert_bias_matrices(progress=0.5)