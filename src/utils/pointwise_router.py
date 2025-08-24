import numpy as np

class PointwiseCorrectionRouter:
    """route pointwise corrections to regions using multiple models"""
    def __init__(self, region_masks, region_centers, model_cls, model_kwargs=None):
        # setup router with masks and models
        self.masks = region_masks
        self.region_centers = region_centers
        if model_kwargs is None:
            model_kwargs = {}
        self.models = {
            name: model_cls(**model_kwargs) for name in region_masks
        }
        self.trained_models = set()

    def fit_all(self, X, y, min_samples=10, **fit_kwargs):
        # fit all region models if we have enough samples
        for name, model in self.models.items():
            mask = self.masks[name]
            count = np.sum(mask)
            if count < min_samples:
                print(f"[{name}] skipped ({count} samples < {min_samples})")
                continue
            print(f"[{name}] training on {count} samples")
            self.trained_models.add(name)
            
            # handle multi-input format
            if isinstance(X, list):
                X_masked = [x[mask] for x in X]
                model.fit(x=X_masked, y=y[mask], **fit_kwargs)
            else:
                model.fit(x=X[mask], y=y[mask], **fit_kwargs)

    def predict(self, X_full, X_coord=None, base=None, sigma=0.15, debug=False):
        # blend predictions using soft routing
        if isinstance(X_full, list):
            N = len(X_full[0])
        else:
            N = len(X_full)
        
        blended = np.zeros((N, 1))
        weight_sum = np.zeros((N, 1))

        for name, model in self.models.items():
            if name not in self.trained_models:
                continue

            if X_coord is None:
                raise ValueError("need X_coord for soft routing")

            mu = np.asarray(self.region_centers[name]).reshape(1, 2)
            d = np.linalg.norm(X_coord - mu, axis=1, keepdims=True)
            w = np.exp(-0.5 * (d / sigma) ** 2)

            # get predictions
            if isinstance(X_full, list):
                delta = model.predict(X_full, batch_size=128)
            else:
                delta = model.predict(X_full, batch_size=128)
                
            blended += w * (base + delta)
            weight_sum += w

            if debug:
                print(f"[{name}] weight sum: {w.sum():.2f}, delta mean: {delta.mean():.6f}")

        eps = 1e-6
        safe_w = np.where(weight_sum < eps, 1.0, weight_sum)
        output = blended / safe_w

        if base is not None:
            fallback_mask = (weight_sum < eps).flatten()
            output[fallback_mask] = base[fallback_mask]
            if debug and np.any(fallback_mask):
                print(f"[router] fallback to base at {np.sum(fallback_mask)} / {len(fallback_mask)} points")

        return output

    def merge_predictions(self, base, delta, debug=False):
        # merge region predictions with rmse scaling
        corrected = np.copy(base).reshape(-1, 1)
        start = 0
        
        # calc global rmse
        if hasattr(self, 'global_residuals'):
            global_rmse = np.sqrt(np.mean(self.global_residuals**2))
        else:
            global_rmse = 0.012  # fallback
        
        for name in self.models:
            if name not in self.trained_models:
                continue

            mask = self.masks[name]
            count = np.sum(mask)
            patch = delta[start:start + count].reshape(-1, 1)

            # rmse proportional scaling
            if hasattr(self, 'region_rmse') and name in self.region_rmse:
                local_rmse = self.region_rmse[name]
                rmse_scale = min(local_rmse / global_rmse, 10.0)
                patch = patch * rmse_scale
                
                if debug:
                    print(f"[{name}] rmse scale: {rmse_scale:.2f} (local: {local_rmse:.4f}, global: {global_rmse:.4f})")
            
            # region specific boosts
            if name == "region_high_gamma":
                patch = patch * 3.0  # strong for high gamma
            elif name == "region_wings":
                patch = patch * 5  # moderate for wings
            elif name == "region_safe":
                patch = patch * 0.8  # reduce in safe zone
                
            if getattr(self.models[name], "use_delta", True):
                corrected[mask] = base[mask] + patch
            else:
                corrected[mask] = patch

            start += count

        return corrected

    def compile_all(self, optimizer_fn):
        # compile all models
        for model in self.models.values():
            model.compile(optimizer=optimizer_fn())

    def save_all(self, path_template):
        # save models to disk
        for name, model in self.models.items():
            model.save(path_template.format(name=name))

    def load_all(self, path_template, custom_objects=None):
        # load models from disk
        from keras.models import load_model
        for name in self.models:
            self.models[name] = load_model(
                path_template.format(name=name),
                custom_objects=custom_objects
            )


import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def _grid_regions(k_vals, tau_vals, bins_k, bins_tau):
    # create grid based regions
    k_edges = np.linspace(k_vals.min(), k_vals.max(), bins_k + 1)
    tau_edges = np.linspace(tau_vals.min(), tau_vals.max(), bins_tau + 1)
    k_idx = np.digitize(k_vals, k_edges) - 1
    tau_idx = np.digitize(tau_vals, tau_edges) - 1
    
    region_masks = {}
    region_centers = {}
    
    for i in range(bins_k):
        for j in range(bins_tau):
            mask = (k_idx == i) & (tau_idx == j)
            if np.any(mask):
                region_masks[f"region_{i}_{j}"] = mask
                region_centers[f"region_{i}_{j}"] = (
                    0.5 * (k_edges[i] + k_edges[i + 1]),
                    0.5 * (tau_edges[j] + tau_edges[j + 1])
                )
    
    return region_masks, region_centers, (k_edges, tau_edges)

def _fixed_grid_regions(k_vals, tau_vals):
    # fixed grid for moneyness and maturity
    atm_band = 0.8
    masks = {
        "region_short_left":  (tau_vals < 0.5) & (k_vals < atm_band),
        "region_short_atm":   (tau_vals < 0.5) & np.isclose(k_vals, atm_band, atol=1e-3),
        "region_short_right": (tau_vals < 0.5) & (k_vals > atm_band),
        "region_mid_left":    (0.5 <= tau_vals) & (tau_vals < 2.0) & (k_vals < 1.0),
        "region_mid_atm":     (0.5 <= tau_vals) & (tau_vals < 2.0) & np.isclose(k_vals, 1.0, atol=1e-3),
        "region_mid_right":   (0.5 <= tau_vals) & (tau_vals < 2.0) & (k_vals > 1.0),
        "region_long_left":   (tau_vals >= 2.0) & (k_vals < atm_band),
        "region_long_atm":    (tau_vals >= 2.0) & np.isclose(k_vals, atm_band, atol=1e-3),
        "region_long_right":  (tau_vals >= 2.0) & (k_vals > atm_band),
    }
    
    region_masks = {}
    region_centers = {}
    
    for name, mask in masks.items():
        region_masks[name] = mask
        region_centers[name] = (np.mean(k_vals[mask]), np.mean(tau_vals[mask]))
    
    return region_masks, region_centers, None

def _hybrid_fixed_adaptive_regions(k_vals, tau_vals, residuals, error_threshold, min_samples, 
                                   atm_lower=0.9, atm_upper=1.1, tau_short=0.5, tau_mid=2.0):
    # hybrid split with fixed bands and adaptive error split for short atm
    region_masks = {}
    region_centers = {}
    
    # maturity bands
    short_mask = tau_vals < tau_short
    mid_mask = (tau_vals >= tau_short) & (tau_vals < tau_mid)
    long_mask = tau_vals >= tau_mid
    
    # moneyness bands
    left_mask = k_vals < atm_lower
    atm_mask = (k_vals >= atm_lower) & (k_vals <= atm_upper)
    right_mask = k_vals > atm_upper
    
    # base regions
    base_masks = {
        "region_short_left":  short_mask & left_mask,
        "region_short_right": short_mask & right_mask,
        "region_mid_left":    mid_mask & left_mask,
        "region_mid_atm":     mid_mask & atm_mask,
        "region_mid_right":   mid_mask & right_mask,
        "region_long_left":   long_mask & left_mask,
        "region_long_atm":    long_mask & atm_mask,
        "region_long_right":  long_mask & right_mask,
    }
    
    # special handling for short atm - our problem area
    short_atm_mask = short_mask & atm_mask
    
    if np.sum(short_atm_mask) >= min_samples * 2:
        # split by error level
        short_atm_residuals = residuals[short_atm_mask]
        high_error_local = np.abs(short_atm_residuals) > error_threshold
        
        if np.sum(high_error_local) >= min_samples and np.sum(~high_error_local) >= min_samples:
            # create global masks
            short_atm_indices = np.where(short_atm_mask)[0]
            
            short_atm_high_mask = np.zeros_like(short_atm_mask, dtype=bool)
            short_atm_low_mask = np.zeros_like(short_atm_mask, dtype=bool)
            
            short_atm_high_mask[short_atm_indices[high_error_local]] = True
            short_atm_low_mask[short_atm_indices[~high_error_local]] = True
            
            region_masks["region_short_atm_high"] = short_atm_high_mask
            region_masks["region_short_atm_low"] = short_atm_low_mask
            
            region_centers["region_short_atm_high"] = (np.mean(k_vals[short_atm_high_mask]), np.mean(tau_vals[short_atm_high_mask]))
            region_centers["region_short_atm_low"] = (np.mean(k_vals[short_atm_low_mask]), np.mean(tau_vals[short_atm_low_mask]))
        else:
            # keep single region if split doesn't work
            region_masks["region_short_atm"] = short_atm_mask
            region_centers["region_short_atm"] = (np.mean(k_vals[short_atm_mask]), np.mean(tau_vals[short_atm_mask]))
    else:
        # keep single if too few samples
        region_masks["region_short_atm"] = short_atm_mask
        region_centers["region_short_atm"] = (np.mean(k_vals[short_atm_mask]), np.mean(tau_vals[short_atm_mask]))
    
    # add other regions
    for name, mask in base_masks.items():
        if np.sum(mask) >= min_samples:
            region_masks[name] = mask
            region_centers[name] = (np.mean(k_vals[mask]), np.mean(tau_vals[mask]))
    
    return region_masks, region_centers, {
        "error_threshold": error_threshold, 
        "atm_bounds": (atm_lower, atm_upper),
        "tau_bounds": (tau_short, tau_mid)
    }

def _atm_term_regions(k_vals, tau_vals):
    # atm and wings regions
    atm_tolerance = 0.05
    atm_mask = np.abs(k_vals - 1.0) <= atm_tolerance
    
    region_masks = {}
    region_centers = {}
    
    if np.any(atm_mask):
        region_masks["region_atm_term"] = atm_mask
        region_centers["region_atm_term"] = (1.0, np.mean(tau_vals[atm_mask]))
        
        wings_mask = ~atm_mask
        if np.any(wings_mask):
            region_masks["region_wings"] = wings_mask  
            region_centers["region_wings"] = (np.mean(k_vals[wings_mask]), np.mean(tau_vals[wings_mask]))
    
    return region_masks, region_centers, atm_tolerance

def _smile_slice_regions(k_vals, tau_vals):
    # regions for each maturity slice
    unique_taus = np.unique(tau_vals)
    region_masks = {}
    region_centers = {}
    
    for i, tau in enumerate(unique_taus):
        mask = np.isclose(tau_vals, tau, atol=1e-3)
        region_masks[f"region_smile_{i}"] = mask
        region_centers[f"region_smile_{i}"] = (np.mean(k_vals[mask]), tau)
    
    return region_masks, region_centers, unique_taus

def _pca_regions(k_vals, tau_vals, residuals, n_clusters):
    # pca then kmeans clustering
    coords = np.stack([k_vals, tau_vals, residuals], axis=1)
    pca = PCA(n_components=2).fit(coords)
    X_proj = pca.transform(coords)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X_proj)
    labels = kmeans.labels_
    
    region_masks = {}
    region_centers = {}
    
    for i in range(n_clusters):
        mask = labels == i
        region_masks[f"region_{i}_0"] = mask
        region_centers[f"region_{i}_0"] = tuple(coords[mask][:, :2].mean(axis=0))
    
    return region_masks, region_centers, pca

def _kmeans_regions(k_vals, tau_vals, residuals, n_clusters):
    # kmeans on all features
    coords = np.stack([k_vals, tau_vals, residuals], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(coords)
    labels = kmeans.labels_
    
    region_masks = {}
    region_centers = {}
    
    for i in range(n_clusters):
        mask = labels == i
        region_masks[f"region_{i}_0"] = mask
        region_centers[f"region_{i}_0"] = tuple(coords[mask][:, :2].mean(axis=0))
    
    return region_masks, region_centers, kmeans

def _spatial_kmeans_regions(k_vals, tau_vals, n_clusters):
    # kmeans on spatial coords only
    coords = np.stack([k_vals, tau_vals], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(coords)
    labels = kmeans.labels_
    
    region_masks = {}
    region_centers = {}
    
    for i in range(n_clusters):
        mask = labels == i
        region_masks[f"region_{i}_0"] = mask
        region_centers[f"region_{i}_0"] = tuple(coords[mask].mean(axis=0))
    
    return region_masks, region_centers, kmeans

def _term_structure_regions(k_vals, tau_vals):
    # term structure bins
    bins = np.linspace(tau_vals.min(), tau_vals.max(), len(np.unique(tau_vals)) + 1)
    tau_idx = np.digitize(tau_vals, bins) - 1
    
    region_masks = {}
    region_centers = {}
    
    for i in range(len(bins) - 1):
        mask = tau_idx == i
        region_masks[f"region_{i}_0"] = mask
        region_centers[f"region_{i}_0"] = (
            np.mean(k_vals[mask]),
            0.5 * (bins[i] + bins[i + 1])
        )
    
    return region_masks, region_centers, bins

def _hierarchical_regions(k_vals, tau_vals, residuals, max_depth, min_samples, improvement_threshold):
    # recursive split by error improvement
    def validate_region_split(region_mask, X_coords, residuals_vals):
        if np.sum(region_mask) < min_samples * 2:
            return False, {}
            
        region_residuals = residuals_vals[region_mask]
        baseline_rmse = np.sqrt(np.mean(region_residuals**2))
        
        region_k = X_coords[region_mask, 0]
        atm_mask_local = np.abs(region_k - 1.0) <= 0.1
        
        if np.sum(atm_mask_local) < min_samples or np.sum(~atm_mask_local) < min_samples:
            return False, {}
            
        atm_rmse = np.sqrt(np.mean(region_residuals[atm_mask_local]**2))
        wings_rmse = np.sqrt(np.mean(region_residuals[~atm_mask_local]**2))
        weighted_rmse = (atm_rmse * np.sum(atm_mask_local) + wings_rmse * np.sum(~atm_mask_local)) / len(region_residuals)
        
        improvement = baseline_rmse - weighted_rmse
        if improvement > improvement_threshold:
            global_atm_mask = np.zeros_like(region_mask, dtype=bool)
            global_wings_mask = np.zeros_like(region_mask, dtype=bool)
            
            region_indices = np.where(region_mask)[0]
            global_atm_mask[region_indices[atm_mask_local]] = True
            global_wings_mask[region_indices[~atm_mask_local]] = True
            
            return True, {"atm": global_atm_mask, "wings": global_wings_mask}
        
        return False, {}
    
    level1_regions = {
        "short_term": tau_vals <= 1.0,
        "long_term": tau_vals > 1.0
    }
    
    final_regions = {}
    X_coords = np.column_stack([k_vals, tau_vals])
    
    def subdivide_region(name, mask, depth=0):
        if depth >= max_depth:
            final_regions[f"{name}_d{depth}"] = mask
            return
            
        should_split, sub_regions = validate_region_split(mask, X_coords, residuals)
        
        if should_split:
            for sub_name, sub_mask in sub_regions.items():
                subdivide_region(f"{name}_{sub_name}", sub_mask, depth + 1)
        else:
            final_regions[f"{name}_d{depth}"] = mask
    
    for region_name, region_mask in level1_regions.items():
        subdivide_region(region_name, region_mask)
    
    region_centers = {}
    for name, mask in final_regions.items():
        if np.any(mask):
            region_centers[name] = (np.mean(k_vals[mask]), np.mean(tau_vals[mask]))
    
    return final_regions, region_centers, {"max_depth": max_depth, "min_samples": min_samples}

def _adaptive_error_regions(k_vals, tau_vals, residuals, error_threshold, min_samples):
    # regions based on error threshold
    high_error_mask = np.abs(residuals) > error_threshold
    low_error_mask = ~high_error_mask
    
    region_masks = {}
    region_centers = {}
    
    if np.sum(high_error_mask) >= min_samples:
        region_masks["region_high_error"] = high_error_mask
        region_centers["region_high_error"] = (np.mean(k_vals[high_error_mask]), np.mean(tau_vals[high_error_mask]))
    
    if np.sum(low_error_mask) >= min_samples:
        region_masks["region_low_error"] = low_error_mask
        region_centers["region_low_error"] = (np.mean(k_vals[low_error_mask]), np.mean(tau_vals[low_error_mask]))
    
    return region_masks, region_centers, error_threshold

def _gamma_based_regions(k_vals, tau_vals, residuals, min_samples):
    # regions based on gamma proxy
    # stronger time decay emphasis
    gamma_proxy = np.exp(-0.5 * ((k_vals - 1.0) / 0.12)**2) / (tau_vals + 0.01)**2.0
    
    wing_mask = (k_vals <= 0.65) | (k_vals >= 1.35)
    atm_zone = ~wing_mask
    
    # very selective - top 30%
    high_gamma_threshold = np.percentile(gamma_proxy, 70)
    
    high_gamma_mask = gamma_proxy >= high_gamma_threshold
    remaining_mask = ~high_gamma_mask
    
    region_masks = {
        "region_high_gamma": high_gamma_mask,
        "region_wings": wing_mask & remaining_mask,
        "region_safe": atm_zone & remaining_mask
    }
    
    region_centers = {}
    for name, mask in region_masks.items():
        if np.sum(mask) >= min_samples:
            region_centers[name] = (np.mean(k_vals[mask]), np.mean(tau_vals[mask]))
    
    return region_masks, region_centers, {"gamma_threshold": high_gamma_threshold, "wing_bounds": [0.75, 1.25]}

def _multi_greek_regions(k_vals, tau_vals, residuals, min_samples):
    # composite greeks sensitivity
    gamma_proxy = np.exp(-0.5 * ((k_vals - 1.0) / 0.2)**2) / np.sqrt(tau_vals)
    vega_proxy = np.sqrt(tau_vals) * np.exp(-0.5 * ((k_vals - 1.0) / 0.3)**2)  
    theta_proxy = 1.0 / (tau_vals + 0.01)  # high for short expiry
    
    # composite score
    sensitivity_score = 0.6 * gamma_proxy + 0.2 * vega_proxy + 0.2 * theta_proxy
    
    # danger zone - atm short term
    atm_short_mask = (np.abs(k_vals - 1.0) <= 0.1) & (tau_vals <= 0.25)
    
    region_masks = {}
    region_centers = {}
    
    if np.sum(atm_short_mask) >= min_samples:
        # split danger zone by sensitivity
        danger_scores = sensitivity_score[atm_short_mask]
        danger_high_threshold = np.percentile(danger_scores, 60)
        
        danger_indices = np.where(atm_short_mask)[0]
        danger_high_local = danger_scores >= danger_high_threshold
        
        danger_ultra_high = np.zeros_like(atm_short_mask, dtype=bool)
        danger_moderate = np.zeros_like(atm_short_mask, dtype=bool)
        
        danger_ultra_high[danger_indices[danger_high_local]] = True
        danger_moderate[danger_indices[~danger_high_local]] = True
        
        region_masks["region_danger_ultra"] = danger_ultra_high 
        region_masks["region_danger_moderate"] = danger_moderate
        
        region_centers["region_danger_ultra"] = (np.mean(k_vals[danger_ultra_high]), np.mean(tau_vals[danger_ultra_high]))
        region_centers["region_danger_moderate"] = (np.mean(k_vals[danger_moderate]), np.mean(tau_vals[danger_moderate]))
    
    # everything else by overall sensitivity
    non_danger_mask = ~atm_short_mask
    if np.sum(non_danger_mask) >= min_samples:
        non_danger_scores = sensitivity_score[non_danger_mask]
        med_threshold = np.percentile(non_danger_scores, 50)
        
        non_danger_indices = np.where(non_danger_mask)[0]
        non_danger_high_local = non_danger_scores >= med_threshold
        
        normal_high = np.zeros_like(non_danger_mask, dtype=bool)
        normal_low = np.zeros_like(non_danger_mask, dtype=bool)
        
        normal_high[non_danger_indices[non_danger_high_local]] = True
        normal_low[non_danger_indices[~non_danger_high_local]] = True
        
        region_masks["region_normal_high"] = normal_high
        region_masks["region_normal_low"] = normal_low
        
        region_centers["region_normal_high"] = (np.mean(k_vals[normal_high]), np.mean(tau_vals[normal_high]))
        region_centers["region_normal_low"] = (np.mean(k_vals[normal_low]), np.mean(tau_vals[normal_low]))
    
    return region_masks, region_centers, {"composite_greeks": True}

def _add_fallback_coverage(region_masks, region_centers, k_vals, tau_vals):
    # add fallback for uncovered points
    if not region_masks:
        return region_masks, region_centers
        
    all_masks = list(region_masks.values())
    combined_mask = np.logical_or.reduce(all_masks)
    uncovered = np.sum(~combined_mask)
    
    if uncovered > 0:
        residual_mask = ~combined_mask
        region_masks["region_fallback_0"] = residual_mask
        region_centers["region_fallback_0"] = (
            np.mean(k_vals[residual_mask]),
            np.mean(tau_vals[residual_mask])
        )
    
    return region_masks, region_centers

def generate_residual_masks(k_vals, tau_vals, residuals, method="grid", bins_k=4, bins_tau=4, 
                            quantile_thresholds=(0.4, 0.8), n_clusters=8, max_depth=3, 
                            min_samples=200, improvement_threshold=0.001, error_threshold=0.015,
                            atm_lower=0.9, atm_upper=1.1, tau_short=0.5, tau_mid=2.0):
    # generate masks using selected method
    method_map = {
        "grid": lambda: _grid_regions(k_vals, tau_vals, bins_k, bins_tau),
        "fixed_grid": lambda: _fixed_grid_regions(k_vals, tau_vals),
        "atm_term_structure": lambda: _atm_term_regions(k_vals, tau_vals),
        "smile_slice": lambda: _smile_slice_regions(k_vals, tau_vals),
        "pca": lambda: _pca_regions(k_vals, tau_vals, residuals, n_clusters),
        "kmeans": lambda: _kmeans_regions(k_vals, tau_vals, residuals, n_clusters),
        "spatial_kmeans": lambda: _spatial_kmeans_regions(k_vals, tau_vals, n_clusters),
        "term_structure": lambda: _term_structure_regions(k_vals, tau_vals),
        "hierarchical": lambda: _hierarchical_regions(k_vals, tau_vals, residuals, max_depth, min_samples, improvement_threshold),
        "adaptive_error": lambda: _adaptive_error_regions(k_vals, tau_vals, residuals, error_threshold, min_samples),
        "hybrid_fixed_adaptive": lambda: _hybrid_fixed_adaptive_regions(k_vals, tau_vals, residuals, error_threshold, min_samples, atm_lower, atm_upper, tau_short, tau_mid),
        "gamma_based": lambda: _gamma_based_regions(k_vals, tau_vals, residuals, min_samples),
        "multi_greek": lambda: _multi_greek_regions(k_vals, tau_vals, residuals, min_samples)
    }
    
    if method not in method_map:
        raise ValueError(f"unsupported method: {method}")
    
    region_masks, region_centers, aux_info = method_map[method]()
    region_masks, region_centers = _add_fallback_coverage(region_masks, region_centers, k_vals, tau_vals)
    
    return region_masks, region_centers, aux_info, method


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_residual_regions(k_vals, tau_vals, residuals, aux_info=None, region_masks=None):
    # viz regions and residuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # check for grid edges
    k_edges = None
    tau_edges = None
    if isinstance(aux_info, tuple) and len(aux_info) == 2:
        k_edges, tau_edges = aux_info
    
    show_heatmap = k_edges is not None and tau_edges is not None

    if show_heatmap:
        bins_k = len(k_edges) - 1
        bins_tau = len(tau_edges) - 1
        rmse_grid = np.full((bins_tau, bins_k), np.nan)
        k_idx = np.digitize(k_vals, k_edges) - 1
        tau_idx = np.digitize(tau_vals, tau_edges) - 1

        for i in range(bins_k):
            for j in range(bins_tau):
                mask = (k_idx == i) & (tau_idx == j)
                if np.any(mask):
                    rmse_grid[j, i] = np.sqrt(np.mean(residuals[mask] ** 2))

        im = ax1.imshow(rmse_grid, origin="lower", cmap="magma", extent=[
            k_edges[0], k_edges[-1], tau_edges[0], tau_edges[-1]
        ], aspect="auto")
        plt.colorbar(im, ax=ax1, label="RMSE")
        ax1.set_title("rmse heatmap")
    else:
        scatter = ax1.scatter(k_vals, tau_vals, c=np.abs(residuals), cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax1)
        ax1.set_title("residual magnitude")

    if region_masks:
        colors = plt.cm.Set3(np.linspace(0, 1, len(region_masks)))
        for idx, (name, mask) in enumerate(region_masks.items()):
            if np.sum(mask) == 0:
                continue
            x = k_vals[mask]
            y = tau_vals[mask]
            
            rect = Rectangle((x.min(), y.min()), 
                           max(x.max()-x.min(), 0.01), 
                           max(y.max()-y.min(), 0.001),
                           fill=False, edgecolor=colors[idx], linewidth=2)
            ax2.add_patch(rect)
            ax2.plot(x.mean(), y.mean(), "o", color=colors[idx], markersize=5)

    for ax in [ax1, ax2]:
        ax.set_xlim(k_vals.min(), k_vals.max())
        ax.set_ylim(tau_vals.min(), tau_vals.max())
        ax.set_xlabel("moneyness (k)")
        ax.set_ylabel("time to maturity (τ)")

    ax2.set_title("rl region overlay")
    plt.tight_layout()
    plt.show()