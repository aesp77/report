from math import tau
import keras
from keras import Model, ops, Sequential
from keras.layers import Dense, LayerNormalization, Activation, Dropout
from keras.metrics import Mean
from keras.saving import register_keras_serializable
import numpy as np

@register_keras_serializable()
class PiecewiseSurfaceDecoderModular(keras.Model):
    def __init__(self, latent_dim, M, K, taus,
                feature_dim=0,
                activation="elu",
                tau_expand=False,
                m_expand=False,
                use_layernorm=False,
                dropout_rate=0.0,
                atm_weighting=False,
                use_moe=False,
                num_experts=4,
                lambda_diversity=0.1,
                atm_specialization=False,
                atm_loss_weight=1.0,
                atm_expert_bias=2.0,
                maturity_experts=2,
                free_experts=2,
                maturity_specialization=True,
                **kwargs):
        super().__init__(**kwargs)
        
        # auto-detect adaptive mode
        self.auto_adaptive = (not maturity_specialization and not atm_specialization)
        
        atm_count = 1 if atm_specialization else 0
        maturity_count = maturity_experts if maturity_specialization else 0
        specialized_experts = atm_count + maturity_count
        if specialized_experts > num_experts:
            raise ValueError(f"Too many specialized experts: ATM({atm_count}) + Maturity({maturity_count}) = {specialized_experts} > Total({num_experts})")
        actual_free_experts = num_experts - specialized_experts
        self.maturity_experts = maturity_count
        self.free_experts = actual_free_experts
        self.atm_experts = atm_count
        
        if self.auto_adaptive:
            print(f"auto-adaptive mode: {num_experts} unbiased experts with learned gating")
        else:
            print(f"expert allocation: ATM={atm_count}, Maturity={maturity_count}, Free={actual_free_experts}, Total={num_experts}")
        
        # Store parameters
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.M = M
        self.K = K
        self.taus = np.array(taus)
        self.activation = activation
        self.tau_expand = tau_expand
        self.m_expand = m_expand
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout_rate
        self.atm_weighting = atm_weighting
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.lambda_diversity = lambda_diversity
        self.atm_specialization = atm_specialization
        self.atm_loss_weight = atm_loss_weight
        self.atm_expert_bias = atm_expert_bias
        self.maturity_specialization = maturity_specialization
        
  

        # Shared dense layers
        self.dense_layers = []
        for units in [256, 128, 64]:
            self.dense_layers.append(Dense(units))
            if self.use_layernorm:
                self.dense_layers.append(LayerNormalization())
            self.dense_layers.append(Activation(self._get_activation(self.activation)))
            if self.dropout_rate > 0:
                self.dense_layers.append(Dropout(self.dropout_rate))
        
        if self.use_moe:
            # MoE output layer
            if self.atm_specialization:
                # ATM-biased gating network
                self.gating_network = Sequential([
                    Dense(32, activation="relu", kernel_regularizer="l2"),
                    Dense(16, activation="relu", kernel_regularizer="l2"),
                    Dense(num_experts)  # No softmax - applied with ATM bias
                ])
            else:
                # Standard gating network
                self.gating_network = Sequential([
                    Dense(32, activation="relu", kernel_regularizer="l2"),
                    Dense(16, activation="relu", kernel_regularizer="l2"),
                    Dense(num_experts, activation="softmax")
                ])
            
            self.experts = []
            for i in range(num_experts):
                if self.atm_specialization and i == 0:
                    # First expert specialized for ATM
                    expert = Sequential([
                        Dense(24, activation="elu", kernel_regularizer="l2"),  # Larger for ATM
                        Dense(1, activation="softplus")
                    ])
                else:
                    # Standard experts
                    expert = Sequential([
                        Dense(16, activation="elu", kernel_regularizer="l2"),
                        Dense(1, activation="softplus")
                    ])
                self.experts.append(expert)
        else:
            # Standard output layer
            self.output_layer = Dense(1, activation="softplus")
    
    def set_previous_decoder_output(self, predictions, ground_truth=None, compute_rmse=True):
        """inject previous decoder's predictions/errors for adaptive biasing
        
        Args:
            predictions: previous decoder output [batch, M, K] or flattened
            ground_truth: true surfaces [batch, M, K] or flattened (optional)
            compute_rmse: whether to compute RMSE stats per region
        """
        # ensure correct shape
        if len(ops.shape(predictions)) == 2:  # if flattened [batch*M*K, 1]
            batch_size = ops.shape(predictions)[0] // (self.M * self.K)
            predictions = ops.reshape(predictions, [batch_size, self.M, self.K])
        
        if ground_truth is not None:
            if len(ops.shape(ground_truth)) == 2:  # if flattened
                batch_size = ops.shape(ground_truth)[0] // (self.M * self.K)
                ground_truth = ops.reshape(ground_truth, [batch_size, self.M, self.K])
            
            # compute errors
            errors = predictions - ground_truth
            self.previous_errors_surface = errors
            
            # compute point-wise rmse for adaptive biasing
            squared_errors = ops.square(errors)
            
            # region-specific RMSE
            self.error_stats = {
                'global_rmse': ops.sqrt(ops.mean(squared_errors)),
                'global_mae': ops.mean(ops.abs(errors)),
            }
            
            # maturity-based errors (along tau axis)
            if compute_rmse:
                # short term (tau < 0.5)
                short_mask = self.taus < 0.5
                if ops.any(short_mask):
                    short_indices = ops.where(short_mask)
                    short_errors = ops.take(squared_errors, short_indices[0], axis=2)
                    self.error_stats['short_rmse'] = ops.sqrt(ops.mean(short_errors))
                
                # medium term (0.5 <= tau < 2.0)
                medium_mask = (self.taus >= 0.5) & (self.taus < 2.0)
                if ops.any(medium_mask):
                    medium_indices = ops.where(medium_mask)
                    medium_errors = ops.take(squared_errors, medium_indices[0], axis=2)
                    self.error_stats['medium_rmse'] = ops.sqrt(ops.mean(medium_errors))
                
                # long term (tau >= 2.0)
                long_mask = self.taus >= 2.0
                if ops.any(long_mask):
                    long_indices = ops.where(long_mask)
                    long_errors = ops.take(squared_errors, long_indices[0], axis=2)
                    self.error_stats['long_rmse'] = ops.sqrt(ops.mean(long_errors))
                
                # strike-based errors
                atm_slice = ops.slice(squared_errors, [0, self.K//2-1, 0], 
                                     [ops.shape(squared_errors)[0], 3, ops.shape(squared_errors)[2]])
                self.error_stats['atm_rmse'] = ops.sqrt(ops.mean(atm_slice))
                
                # wings
                left_wing = ops.slice(squared_errors, [0, 0, 0], 
                                     [ops.shape(squared_errors)[0], 2, ops.shape(squared_errors)[2]])
                right_wing = ops.slice(squared_errors, [0, self.K-2, 0],
                                      [ops.shape(squared_errors)[0], 2, ops.shape(squared_errors)[2]])
                wing_errors = ops.concatenate([left_wing, right_wing], axis=1)
                self.error_stats['wing_rmse'] = ops.sqrt(ops.mean(wing_errors))
            
            # store flattened errors for point-wise use during training
            self.previous_errors_flat = ops.reshape(errors, [-1, 1])
            
            print(f"injected previous decoder errors - global RMSE: {float(self.error_stats['global_rmse']):.4f}")
            if compute_rmse and 'short_rmse' in self.error_stats:
                print(f"  short-term RMSE: {float(self.error_stats.get('short_rmse', 0)):.4f}")
                print(f"  long-term RMSE: {float(self.error_stats.get('long_rmse', 0)):.4f}")
        else:
            # just store predictions if no ground truth
            self.previous_predictions = predictions
            print("stored previous decoder predictions (no ground truth provided)")
    
    def clear_previous_decoder_output(self):
        """clear any stored previous decoder information"""
        if hasattr(self, 'previous_errors_surface'):
            del self.previous_errors_surface
        if hasattr(self, 'previous_errors_flat'):
            del self.previous_errors_flat
        if hasattr(self, 'error_stats'):
            del self.error_stats
        if hasattr(self, 'previous_predictions'):
            del self.previous_predictions
        print("cleared previous decoder information")

    def build_training_data_from_surfaces(self, Z_latent, Y_surface_flat, strike_tensor, tau_tensor, F_features=None):
        # convert to numpy using Keras 3 ops
        Z_latent = ops.convert_to_numpy(Z_latent)
        Y_surface_flat = ops.convert_to_numpy(Y_surface_flat)
        strike_tensor = ops.convert_to_numpy(strike_tensor)
        tau_tensor = ops.convert_to_numpy(tau_tensor)
        if F_features is not None:
            F_features = ops.convert_to_numpy(F_features)
        
        # rest of method unchanged
        M, K = self.M, self.K
        m_grid, tau_grid = np.meshgrid(strike_tensor, tau_tensor)
        m_flat = m_grid.reshape(-1, 1)
        tau_flat = tau_grid.reshape(-1, 1)

        X_f, X_m, X_tau, y = [], [], [], []
        for i, (z_vec, surf_flat) in enumerate(zip(Z_latent, Y_surface_flat)):
            if self.feature_dim > 0 and F_features is not None:
                zf = np.concatenate([z_vec, F_features[i]], axis=-1)
            else:
                zf = z_vec
            z_repeat = np.repeat(zf[None, :], M * K, axis=0)
            X_f.append(z_repeat)
            X_m.append(m_flat)
            X_tau.append(tau_flat)
            y.append(surf_flat.reshape(-1, 1))

        return [np.vstack(X_f), np.vstack(X_m), np.vstack(X_tau)], np.vstack(y)
    
    def _get_activation(self, name):
        return {
            "relu": keras.activations.relu,
            "elu": keras.activations.elu,
            "gelu": keras.activations.gelu,
            "tanh": keras.activations.tanh,
            "sigmoid": keras.activations.sigmoid
        }.get(name, keras.activations.elu)

    def _compute_atm_proximity(self, m):
        """Compute how close moneyness is to ATM (m=1.0)"""
        return ops.exp(-5.0 * ops.square(m - 1.0))

    def call(self, inputs, training=None):
        if self.feature_dim > 0 and len(inputs) == 4:
            z, m, tau, f = inputs
            z = ops.concatenate([z, f], axis=-1)
        else:
            z, m, tau = inputs

        if self.m_expand:
            m = ops.concatenate([m, ops.square(m)], axis=-1)
        if self.tau_expand:
            tau = ops.concatenate([tau, ops.square(tau), ops.log(ops.add(tau, 1e-3))], axis=-1)

        x = ops.concatenate([z, m, tau], axis=-1)
        
        # Store original inputs for specialization
        if self.use_moe:
            m_orig = inputs[1] if len(inputs) > 1 else m
            tau_orig = inputs[2] if len(inputs) > 2 else tau
            self.current_moneyness = m_orig
            self.current_tau = tau_orig
        
        # Shared feature processing
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        if self.use_moe:
            # Get raw gate logits
            gate_logits = self.gating_network(x, training=training)
            
            # Apply specialization biases
            if self.atm_specialization or self.maturity_specialization or self.auto_adaptive or hasattr(self, 'error_stats'):
                expert_biases = self._compute_expert_biases(m_orig, tau_orig)
                gate_logits = gate_logits + expert_biases
            
            gate_weights = ops.softmax(gate_logits)
            
            expert_outputs = []
            for expert in self.experts:
                output = expert(x, training=training)
                expert_outputs.append(output)
            
            # Store for diversity penalty
            if training:
                self.last_expert_outputs = expert_outputs
            
            # Weighted combination
            expert_stack = ops.stack(expert_outputs, axis=-1)
            gate_expanded = ops.expand_dims(gate_weights, axis=1)
            weighted_output = ops.sum(expert_stack * gate_expanded, axis=-1)
            
            return weighted_output
        else:
            return self.output_layer(x, training=training)

    def compute_moe_loss(self, inputs, y_true, training=True):
        """Custom loss computation with MoE diversity penalty and ATM weighting"""
        y_pred = self(inputs, training=training)
        
        # Base MSE loss with optional ATM weighting
        if self.atm_loss_weight > 1.0 and len(inputs) > 1:
            # Weight ATM points higher
            m_orig = inputs[1]
            atm_proximity = self._compute_atm_proximity(m_orig)
            loss_weights = 1.0 + (self.atm_loss_weight - 1.0) * atm_proximity
            weighted_errors = loss_weights * ops.square(y_pred - y_true)
            mse_loss = ops.mean(weighted_errors)
        else:
            # Standard MSE
            mse_loss = ops.mean(ops.square(y_pred - y_true))
        
        if self.use_moe and self.lambda_diversity > 0.0 and training:
            # Diversity penalty
            diversity_loss = 0.0
            if hasattr(self, 'last_expert_outputs'):
                count = 0
                for i in range(self.num_experts):
                    for j in range(i+1, self.num_experts):
                        correlation = ops.mean(self.last_expert_outputs[i] * self.last_expert_outputs[j])
                        diversity_loss += ops.square(correlation)
                        count += 1
                
                if count > 0:
                    diversity_loss = diversity_loss / count
            
            total_loss = mse_loss + self.lambda_diversity * diversity_loss
            return total_loss
        
        return mse_loss

    def predict_surface(self, z, f, m_flat, tau_flat):
        # convert to tensors for prediction
        z = ops.convert_to_tensor(z, dtype="float32")
        if len(ops.shape(z)) == 1:
            z = ops.reshape(z, [1, -1])
        
        if self.feature_dim > 0:
            f = ops.convert_to_tensor(f, dtype="float32")
            if len(ops.shape(f)) == 1:
                f = ops.reshape(f, [1, -1])
            zf = ops.concatenate([z, f], axis=-1)
        else:
            zf = z

        z_repeat = ops.repeat(zf, self.M * self.K, axis=0)
        m_tensor = ops.convert_to_tensor(m_flat, dtype="float32")
        tau_tensor = ops.convert_to_tensor(tau_flat, dtype="float32")
        
        iv = self([z_repeat, m_tensor, tau_tensor], training=False)
        return ops.reshape(iv, [self.M, self.K])

    def analyze_gating(self, inputs):
        """Analyze expert gating patterns (only for MoE mode)"""
        if not self.use_moe:
            return None
        
        z, m, tau = inputs[:3]

        if self.m_expand:
            m = ops.concatenate([m, ops.square(m)], axis=-1)
        if self.tau_expand:
            tau = ops.concatenate([tau, ops.square(tau), ops.log(ops.add(tau, 1e-3))], axis=-1)

        x = ops.concatenate([z, m, tau], axis=-1)
        
        for layer in self.dense_layers:
            x = layer(x, training=False)
        
        # Handle specialization in analysis
        gate_logits = self.gating_network(x, training=False)
        if self.atm_specialization or self.maturity_specialization or self.auto_adaptive or hasattr(self, 'error_stats'):
            m_orig = inputs[1]
            tau_orig = inputs[2]
            expert_biases = self._compute_expert_biases(m_orig, tau_orig)
            gate_logits = gate_logits + expert_biases
        
        gate_weights = ops.softmax(gate_logits)
        
        # Convert everything to numpy for analysis
        analysis = {
            'gate_weights': ops.convert_to_numpy(gate_weights),
            'expert_activations': ops.convert_to_numpy(ops.mean(gate_weights, axis=0)),
            'gate_entropy': ops.convert_to_numpy(-ops.sum(gate_weights * ops.log(gate_weights + 1e-8), axis=1)),
            'dominant_expert': ops.convert_to_numpy(ops.argmax(gate_weights, axis=1)),
        }
        
        return analysis

    def refine_surface(self, surface, recent_surfaces=None, lambda_cal=0.1, lambda_smile=0.1, lambda_history=0.05):
        import scipy.optimize
        M, K = self.M, self.K
        flat = surface.flatten()
        
        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            
            # Fixed calendar penalty: total variance σ²T must be non-decreasing
            total_var = np.square(surf) * self.taus[:, None]
            cal_penalty = np.mean(np.clip(-np.diff(total_var, axis=0), 0, None))
            
            # Smile penalty
            smile_diff2 = np.diff(surf, n=2, axis=1)
            if self.atm_weighting:
                weights = 1-np.abs(np.arange(K) - K // 2) / (K // 2)
                weighted_smile = (smile_diff2 ** 2) * weights[1:-1][None, :]
                smile_pen = np.mean(weighted_smile)
            else:
                smile_pen = np.mean(np.clip(-smile_diff2, 0, None))
            
            total_loss = lambda_cal * cal_penalty + lambda_smile * smile_pen
            
            # Historical pattern matching if provided
            if recent_surfaces is not None:
                # ATM term structure matching
                atm_current = surf[:, K//2]
                atm_recent = recent_surfaces[:, :, K//2].mean(axis=0)
                atm_penalty = np.mean((atm_current - atm_recent)**2)
                
                # Short-term behavior (τ < 0.55)
                short_mask = self.taus < 0.55
                if np.any(short_mask):
                    short_current = surf[short_mask, :]
                    short_recent = recent_surfaces[:, short_mask, :].mean(axis=0)
                    short_penalty = np.mean((short_current - short_recent)**2)
                else:
                    short_penalty = lambda_cal
                
                # Wing behavior (extreme deltas)
                wing_indices = [0, 1, K-2, K-1]  # Far OTM strikes
                wing_current = surf[:, wing_indices]
                wing_recent = recent_surfaces[:, :, wing_indices].mean(axis=0)
                wing_penalty = np.mean((wing_current - wing_recent)**2)
                
                total_loss += lambda_history * (atm_penalty + short_penalty + wing_penalty)
            
            return total_loss
        
        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)
    
    def _compute_adaptive_biases(self, m, tau):
        """compute adaptive biases based on previous decoder errors"""
        batch_size = ops.shape(m)[0]
        
        # initialize adaptive bias network if needed
        if not hasattr(self, 'adaptive_bias_net'):
            self.adaptive_bias_net = Sequential([
                Dense(64, activation='relu', kernel_initializer='orthogonal'),
                Dense(32, activation='relu', kernel_initializer='orthogonal'),
                Dense(self.num_experts, kernel_initializer='zeros')
            ])
        
        # convert inputs to tensors if they're numpy
        m = ops.convert_to_tensor(m) if isinstance(m, np.ndarray) else m
        tau = ops.convert_to_tensor(tau) if isinstance(tau, np.ndarray) else tau
        
        # create soft structural priors
        structural_priors = []
        
        # distribute experts across tau and strike dimensions
        tau_centers = [0.1, 1.0, 2.0, 4.0]  # 4 tau regions
        strike_centers = [0.85, 1.0, 1.15]  # 3 strike regions
        
        expert_idx = 0
        for i in range(min(4, self.num_experts)):  # tau-focused experts
            tau_center = ops.convert_to_tensor(tau_centers[i], dtype='float32')
            tau_affinity = ops.exp(-ops.square((tau - tau_center) / 0.5))
            structural_priors.append(tau_affinity)
            expert_idx += 1
        
        for i in range(min(3, self.num_experts - expert_idx)):  # strike-focused experts
            strike_center = ops.convert_to_tensor(strike_centers[i], dtype='float32')
            strike_affinity = ops.exp(-ops.square((m - strike_center) / 0.15))
            structural_priors.append(strike_affinity)
            expert_idx += 1
        
        # any remaining experts get uniform prior
        while len(structural_priors) < self.num_experts:
            structural_priors.append(ops.ones((batch_size, 1)) * 0.5)
        
        structural_prior = ops.concatenate(structural_priors, axis=-1)
        
        # normalize priors
        structural_prior = structural_prior / (ops.sum(structural_prior, axis=-1, keepdims=True) + 1e-8)
        
        # base features for adaptive network
        features = ops.concatenate([
            m,
            tau,
            ops.log(tau + 0.01),
            ops.abs(m - 1.0),
            ops.square(m - 1.0),
            ops.exp(-ops.square((tau - 2.0)/0.5)),
            ops.exp(-ops.square((tau - 3.0)/1.0)),
        ], axis=-1)
        
        # add error-based features if available
        if hasattr(self, 'error_stats'):
            error_features = []
            
            if 'long_rmse' in self.error_stats:
                long_error_signal = ops.ones((batch_size, 1)) * self.error_stats['long_rmse']
                error_features.append(long_error_signal)
            
            if 'short_rmse' in self.error_stats:
                short_error_signal = ops.ones((batch_size, 1)) * self.error_stats['short_rmse']
                error_features.append(short_error_signal)
            
            if 'atm_rmse' in self.error_stats:
                atm_error_signal = ops.ones((batch_size, 1)) * self.error_stats['atm_rmse']
                error_features.append(atm_error_signal)
            
            if error_features:
                error_context = ops.concatenate(error_features, axis=-1)
                features = ops.concatenate([features, error_context], axis=-1)
        
        # compute learned adjustments
        learned_adjustments = self.adaptive_bias_net(features)
        
        # combine structural prior with learned adjustments
        alpha = 0.7  # how much to trust the prior vs learned
        adaptive_biases = structural_prior * alpha + ops.softmax(learned_adjustments) * (1 - alpha)
        
        # scale up the biases
        adaptive_biases = adaptive_biases * 5.0
        
        # apply error-based boosting if available
        if hasattr(self, 'error_stats') and 'long_rmse' in self.error_stats:
            long_term_mask = tau > 2.0
            if float(self.error_stats.get('long_rmse', 0)) > 0.02:
                boost_factor = 1.0 + (float(self.error_stats['long_rmse']) - 0.02) * 10.0
                # simpler boosting - just multiply certain experts
                for i in range(3, min(6, self.num_experts)):
                    adaptive_biases = ops.where(
                        ops.expand_dims(long_term_mask, -1),
                        ops.concat([
                            adaptive_biases[:, :i],
                            adaptive_biases[:, i:i+1] * boost_factor,
                            adaptive_biases[:, i+1:]
                        ], axis=-1) if i < self.num_experts - 1 else adaptive_biases,
                        adaptive_biases
                    )
        
        return adaptive_biases

    def _compute_expert_biases(self, m, tau):
        batch_size = ops.shape(m)[0]
        
        #  if no specialization, use adaptive biases
        if self.auto_adaptive:
            return self._compute_adaptive_biases(m, tau)
        
        #  when previous decoder errors available
        if hasattr(self, 'error_stats'):
            return self._compute_adaptive_biases(m, tau)
        
        # FIXED MODE: original implementation
        biases = ops.zeros((batch_size, self.num_experts))
        
        # annealing logic
        if hasattr(self, '_training_step'):
            self._training_step += 1
        else:
            self._training_step = 0
        
        if not hasattr(self, '_total_training_steps'):
            self._total_training_steps = 2000
        
        annealing_fraction = getattr(self, 'annealing_fraction', 0.35)
        annealing_steps = int(self._total_training_steps * annealing_fraction)
        
        progress = min(1.0, self._training_step / annealing_steps)
        max_bias_start = getattr(self, 'max_bias_start', 12.0)
        max_bias_end = getattr(self, 'max_bias_end', 4.0)
        moderate_bias_start = getattr(self, 'moderate_bias_start', 4.0)
        moderate_bias_end = getattr(self, 'moderate_bias_end', 1.0)
        
        max_bias = max_bias_end + (max_bias_start - max_bias_end) * (1.0 - progress)
        moderate_bias = moderate_bias_end + (moderate_bias_start - moderate_bias_end) * (1.0 - progress)
        
        if self.maturity_specialization:
            tau_bands = [0.3, 0.75, 2, 3.5]
            bias_strengths = [
                max_bias*1.5,
                moderate_bias+1.0,
                moderate_bias + 2.0,
                moderate_bias + 1.5,
                moderate_bias + 4
            ]
            
            for i in range(self.maturity_experts):
                if i == 0:
                    condition = tau < tau_bands[0]
                    bias_strength = bias_strengths[0]
                elif i == 1:
                    condition = (tau >= tau_bands[0]) & (tau < tau_bands[1])
                    bias_strength = bias_strengths[1]
                elif i == 2:
                    condition = (tau >= tau_bands[1]) & (tau < tau_bands[2])
                    bias_strength = bias_strengths[2]
                else:
                    condition = tau >= tau_bands[2]
                    bias_strength = bias_strengths[3]
                
                maturity_bias = ops.where(condition, bias_strength, 0.0)
                biases = ops.scatter_update(biases, ops.stack([ops.arange(batch_size), 
                                          ops.full([batch_size], i)], axis=1),
                                          ops.squeeze(maturity_bias, axis=-1))
        
        # strike specialization
        remaining_experts = self.num_experts - self.maturity_experts
        if remaining_experts > 0:
            strike_bands = [0.9, 1.1]
            
            for i in range(min(remaining_experts, 3)):
                expert_idx = self.maturity_experts + i
                
                if i == 0:  # ITM
                    condition = m < strike_bands[0]
                    bias_strength = max_bias * 1.5 * (1 - progress)
                elif i == 1:  # ATM
                    condition = (m >= strike_bands[0]) & (m <= strike_bands[1])
                    calendar_strength = max_bias * 1.5 * (1.0 - progress)
                    
                    ultra_short_boost = ops.where(tau < 0.3, calendar_strength, 0.0)
                    short_boost = ops.where((tau >= 0.3) & (tau <0.85), moderate_bias, 0.0)
                    medium_boost = ops.where((tau >= 0.85) & (tau < 2), 2.0, 0.0)
                    long_boost = ops.where(tau >= 2, moderate_bias, 0.0)
                    
                    bias_strength = 1.0 + ultra_short_boost*2 + short_boost + medium_boost + long_boost
                else:  # OTM
                    condition = m > strike_bands[1]
                    far_otm = m > 1.3
                    bias_strength = ops.where(far_otm,
                                         max_bias * 5.0 * (1 - progress),
                                         max_bias * 2.5 * (1 - progress))
                
                strike_bias = ops.where(condition, bias_strength, 0.0)
                biases = ops.scatter_update(biases, ops.stack([ops.arange(batch_size),
                                          ops.full([batch_size], expert_idx)], axis=1),
                                          ops.squeeze(strike_bias, axis=-1))
        
        return biases

    def get_config(self):
        # convert taus back to list for serialization
        taus_list = self.taus.numpy().tolist() if hasattr(self.taus, 'numpy') else self.taus
        return {
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "M": self.M,
            "K": self.K,
            "taus": taus_list,
            "activation": self.activation,
            "tau_expand": self.tau_expand,
            "m_expand": self.m_expand,
            "use_layernorm": self.use_layernorm,
            "dropout_rate": self.dropout_rate,
            "atm_weighting": self.atm_weighting,
            "use_moe": self.use_moe,
            "num_experts": self.num_experts,
            "lambda_diversity": self.lambda_diversity,
            "atm_specialization": self.atm_specialization,
            "atm_loss_weight": self.atm_loss_weight,
            "atm_expert_bias": self.atm_expert_bias,
            "maturity_experts": self.maturity_experts,
            "free_experts": self.free_experts,
            "maturity_specialization": self.maturity_specialization
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)