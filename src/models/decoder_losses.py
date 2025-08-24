import numpy as np
import keras
from keras import ops
from keras.saving import register_keras_serializable

def create_black_scholes_hybrid_loss(taus, rel_strikes, M, K, mse_weight=0.5, 
                                  vega_weight=0.3, gamma_weight=0.2, 
                                  S0=100, r=0.02, base_vol=0.2):
   # hybrid loss combining mse with vega and gamma weights
   strike_grid, tau_grid = np.meshgrid(rel_strikes, taus)
   K_abs = strike_grid * S0
   
   # bs d1 calc
   d1 = np.clip((np.log(S0/K_abs) + (r + 0.5*base_vol**2)*tau_grid) / (base_vol*np.sqrt(tau_grid)), -10, 10)
   phi_d1 = np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
   
   # vega and gamma
   raw_vega = S0 * np.sqrt(tau_grid) * phi_d1
   raw_gamma = phi_d1 / (S0 * (base_vol**2) * tau_grid)
   
   # convert to weights
   vega_weights = raw_vega.flatten()
   gamma_weights = raw_gamma.flatten()
   
   # normalize
   vega_weights = vega_weights / vega_weights.mean()
   gamma_weights = gamma_weights / gamma_weights.mean()
       
   @register_keras_serializable()
   def black_scholes_loss(y_true, y_pred):
       squared_errors = ops.square(y_true - y_pred)
       
       # base mse
       mse_loss = ops.mean(squared_errors)
       
       # vega weighting
       vega_tensor = ops.convert_to_tensor(vega_weights, dtype=y_pred.dtype)
       vega_tensor = ops.reshape(vega_tensor, (1, -1))
       vega_loss = ops.mean(squared_errors * vega_tensor)
       
       # gamma weighting  
       gamma_tensor = ops.convert_to_tensor(gamma_weights, dtype=y_pred.dtype)
       gamma_tensor = ops.reshape(gamma_tensor, (1, -1))
       gamma_loss = ops.mean(squared_errors * gamma_tensor)
       
       # combine
       total_loss = (mse_weight * mse_loss + 
                    vega_weight * vega_loss + 
                    gamma_weight * gamma_loss)
       
       return total_loss * 100
   
   return black_scholes_loss, vega_weights, gamma_weights

def create_mse_vega_hybrid_loss(taus, rel_strikes, M, K, mse_weight=0.6, vega_weight=0.4, scale=100):
   # combine mse with vega weighting for critical regions
   tau_grid, strike_grid = np.meshgrid(rel_strikes, taus)
   K_abs = strike_grid * 100
   
   d1 = (np.log(100/K_abs) + 0.02*tau_grid) / (0.2*np.sqrt(tau_grid))
   phi_d1 = np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
   
   vega_weights = 100 * np.sqrt(tau_grid) * phi_d1
   vega_weights = vega_weights.flatten()
   vega_weights = vega_weights / vega_weights.mean()
   
   print(f"Vega weight range: {vega_weights.min():.3f} to {vega_weights.max():.3f}")
   
   def hybrid_loss(y_true, y_pred):
       squared_errors = ops.square(y_true - y_pred)
       
       # mse component
       mse_loss = ops.mean(squared_errors)
       
       # vega component
       weights_tensor = ops.convert_to_tensor(vega_weights, dtype=y_pred.dtype)
       weights_tensor = ops.reshape(weights_tensor, (1, -1))
       vega_loss = ops.mean(squared_errors * weights_tensor)
       
       # weighted combination
       total_loss = scale * (mse_weight * mse_loss + vega_weight * vega_loss)
       return total_loss
   
   return hybrid_loss, vega_weights

def create_mse_gamma_hybrid_loss(taus, rel_strikes, M, K, mse_weight=0.7, gamma_weight=0.3):
   # mse plus gamma weighting for hedging sensitivity
   tau_grid, strike_grid = np.meshgrid(rel_strikes, taus)
   K_abs = strike_grid * 100
   
   # gamma higher for atm short term
   d1 = (np.log(100/K_abs) + 0.02*tau_grid) / (0.2*np.sqrt(tau_grid))
   phi_d1 = np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
   
   gamma_weights = phi_d1 / (100 * 0.2 * np.sqrt(tau_grid))
   gamma_weights = gamma_weights.flatten()
   gamma_weights = gamma_weights / gamma_weights.mean()
   
   def hybrid_loss(y_true, y_pred):
       squared_errors = ops.square(y_true - y_pred)
       
       mse_loss = ops.mean(squared_errors)
       
       weights_tensor = ops.convert_to_tensor(gamma_weights, dtype=y_pred.dtype)
       weights_tensor = ops.reshape(weights_tensor, (1, -1))
       gamma_loss = ops.mean(squared_errors * weights_tensor)
       
       return mse_weight * mse_loss + gamma_weight * gamma_loss
   
   return hybrid_loss, gamma_weights

def create_vega_weighted_loss(taus, rel_strikes, M, K, S0=100, r=0.02, min_weight=0.5):
   # weight errors by option vega with floor
   tau_grid, strike_grid = np.meshgrid(rel_strikes, taus)
   K_abs = strike_grid * S0
   
   # calc vega weights
   d1 = (np.log(S0/K_abs) + r*tau_grid) / (0.2*np.sqrt(tau_grid))
   phi_d1 = np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
   
   vega_weights = S0 * np.sqrt(tau_grid) * phi_d1
   vega_weights = vega_weights.flatten()
   vega_weights = vega_weights / vega_weights.mean()
   
   # apply minimum weight floor
   vega_weights = np.maximum(vega_weights, min_weight)
   
   print(f"Weight range: {vega_weights.min():.3f} to {vega_weights.max():.3f}")
   
   def vega_loss(y_true, y_pred):
       squared_errors = ops.square(y_true - y_pred)
       weights_tensor = ops.convert_to_tensor(vega_weights, dtype=y_pred.dtype)
       weights_tensor = ops.reshape(weights_tensor, (1, -1))
       weighted_errors = squared_errors * weights_tensor
       return ops.mean(weighted_errors)
   
   return vega_loss, vega_weights

def create_gamma_weighted_loss(taus, rel_strikes, M, K, S0=100, r=0.02):
   # weight by gamma for hedging frequency
   tau_grid, strike_grid = np.meshgrid(rel_strikes, taus)
   K_abs = strike_grid * S0
   
   # gamma approx
   d1 = (np.log(S0/K_abs) + r*tau_grid) / (0.2*np.sqrt(tau_grid))
   phi_d1 = np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
   
   gamma_weights = phi_d1 / (S0 * 0.2 * np.sqrt(tau_grid))
   gamma_weights = gamma_weights.flatten()
   gamma_weights = gamma_weights / gamma_weights.mean()
   
   def gamma_loss(y_true, y_pred):
       squared_errors = ops.square(y_true - y_pred)
       weighted_errors = squared_errors * gamma_weights[None, :]
       return ops.mean(weighted_errors)
   
   return gamma_loss, gamma_weights

def create_pnl_impact_loss(taus, rel_strikes, M, K, portfolio_delta=1000):
   # weight by pnl impact of vol errors
   tau_grid, strike_grid = np.meshgrid(rel_strikes, taus)
   
   # pnl impact based on position size times vega
   position_weights = np.exp(-1.0 * tau_grid) * np.exp(-4.0 * (strike_grid - 1.0)**2)
   
   # vega multiplier
   d1 = (np.log(1.0/strike_grid)) / (0.2*np.sqrt(tau_grid))
   phi_d1 = np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
   vega_mult = np.sqrt(tau_grid) * phi_d1
   
   pnl_weights = position_weights * vega_mult * portfolio_delta
   pnl_weights = pnl_weights.flatten()
   pnl_weights = pnl_weights / pnl_weights.mean()
   
   def pnl_loss(y_true, y_pred):
       squared_errors = ops.square(y_true - y_pred)
       weighted_errors = squared_errors * pnl_weights[None, :]
       return ops.mean(weighted_errors)
   
   return pnl_loss, pnl_weights