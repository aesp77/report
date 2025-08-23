from math import tau
import keras
from keras import Model, ops, Sequential
from keras.layers import Dense, LayerNormalization, Activation, Dropout
from keras.metrics import Mean
from keras.saving import register_keras_serializable
import numpy as np

@register_keras_serializable()
class PiecewiseSurfaceDecoderModular(keras.Model):
    """Modular piecewise decoder with Mixture-of-Experts (MoE) for volatility surface.
    Hidden layers: Dense with configurable activation (elu, relu, gelu, tanh, sigmoid).
    MoE options:
        use_moe: enable mixture-of-experts
        num_experts: number of experts
        atm_specialization: ATM expert bias
        maturity_specialization: maturity expert bias
        lambda_diversity: diversity penalty weight
    Output: Dense(1) with softplus activation.
    """
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
        atm_count = 1 if atm_specialization else 0
        maturity_count = maturity_experts if maturity_specialization else 0
        specialized_experts = atm_count + maturity_count
        if specialized_experts > num_experts:
            raise ValueError(f"Too many specialized experts: ATM({atm_count}) + Maturity({maturity_count}) = {specialized_experts} > Total({num_experts})")
        actual_free_experts = num_experts - specialized_experts
        self.maturity_experts = maturity_count
        self.free_experts = actual_free_experts
        self.atm_experts = atm_count
        # Model parameters
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
        # Shared dense layers for feature processing
        self.dense_layers = []
        for units in [256, 128, 64]:
            self.dense_layers.append(Dense(units))
            if self.use_layernorm:
                self.dense_layers.append(LayerNormalization())
            self.dense_layers.append(Activation(self._get_activation(self.activation)))
            if self.dropout_rate > 0:
                self.dense_layers.append(Dropout(self.dropout_rate))
        # Mixture-of-Experts output layer setup
        if self.use_moe:
            if self.atm_specialization:
                self.gating_network = Sequential([
                    Dense(32, activation="relu", kernel_regularizer="l2"),
                    Dense(16, activation="relu", kernel_regularizer="l2"),
                    Dense(num_experts)
                ])
            else:
                self.gating_network = Sequential([
                    Dense(32, activation="relu", kernel_regularizer="l2"),
                    Dense(16, activation="relu", kernel_regularizer="l2"),
                    Dense(num_experts, activation="softmax")
                ])
            self.experts = []
            for i in range(num_experts):
                if self.atm_specialization and i == 0:
                    expert = Sequential([
                        Dense(24, activation="elu", kernel_regularizer="l2"),
                        Dense(1, activation="softplus")
                    ])
                else:
                    expert = Sequential([
                        Dense(16, activation="elu", kernel_regularizer="l2"),
                        Dense(1, activation="softplus")
                    ])
                self.experts.append(expert)
        else:
            self.output_layer = Dense(1, activation="softplus")
            
    def build_training_data_from_surfaces(self, Z_latent, Y_surface_flat, strike_tensor, tau_tensor, F_features=None):
        M, K = self.M, self.K
        m_grid, tau_grid = np.meshgrid(strike_tensor, tau_tensor)
        m_flat = m_grid.reshape(-1, 1)
        tau_flat = tau_grid.reshape(-1, 1)

        X_f, X_m, X_tau, y = [], [], [], []
        for i, (z_vec, surf_flat) in enumerate(zip(Z_latent, Y_surface_flat)):
            if self.feature_dim > 0 and F_features is not None:
                zf = np.concatenate([z_vec, F_features[i]], axis=-1)  # Changed this line
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
            """Forward pass for surface decoder.
            If MoE enabled, combines expert outputs using gating network and specialization biases.
            """
            if self.feature_dim > 0 and len(inputs) == 4:
                z, m, tau, f = inputs
                z = ops.concatenate([z, f], axis=-1)
            else:
                z, m, tau = inputs
            # Optionally expand moneyness and tau features
            if self.m_expand:
                m = ops.concatenate([m, ops.square(m)], axis=-1)
            if self.tau_expand:
                tau = ops.concatenate([tau, ops.square(tau), ops.log(ops.add(tau, 1e-3))], axis=-1)
            x = ops.concatenate([z, m, tau], axis=-1)
            # Shared feature processing
            for layer in self.dense_layers:
                x = layer(x, training=training)
            if self.use_moe:
                # MoE: gating network and expert outputs
                m_orig = inputs[1] if len(inputs) > 1 else m
                tau_orig = inputs[2] if len(inputs) > 2 else tau
                self.current_moneyness = m_orig
                self.current_tau = tau_orig
                gate_logits = self.gating_network(x, training=training)
                if self.atm_specialization or self.maturity_specialization:
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
                # Weighted combination of expert outputs
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
        z = ops.convert_to_numpy(z).reshape(1, -1).astype(np.float32)
        if self.feature_dim > 0:
            f = ops.convert_to_numpy(f).reshape(1, -1).astype(np.float32)
            zf = np.concatenate([z, f], axis=-1)
        else:
            zf = z

        z_repeat = np.repeat(zf, self.M * self.K, axis=0)
        m_tensor = ops.convert_to_tensor(m_flat.astype("float32"))
        tau_tensor = ops.convert_to_tensor(tau_flat.astype("float32"))
        
        iv = self([z_repeat, m_tensor, tau_tensor], training=False)
        return ops.convert_to_numpy(iv).reshape(self.M, self.K)

    def analyze_gating(self, inputs):
        """Analyze expert gating patterns (only for MoE mode)"""
        if not self.use_moe:
            return None
        
        # X_zf already contains latent + features concatenated, so only use first 3 inputs
        z, m, tau = inputs[:3]

        if self.m_expand:
            m = ops.concatenate([m, ops.square(m)], axis=-1)
        if self.tau_expand:
            tau = ops.concatenate([tau, ops.square(tau), ops.log(ops.add(tau, 1e-3))], axis=-1)

        x = ops.concatenate([z, m, tau], axis=-1)
        
        for layer in self.dense_layers:
            x = layer(x, training=False)
        
        # Handle ATM specialization in analysis
        if self.atm_specialization:
            m_orig = inputs[1]
            atm_proximity = self._compute_atm_proximity(m_orig)
            gate_logits = self.gating_network(x, training=False)
            atm_bias = self.atm_expert_bias * atm_proximity
            atm_bias_reshaped = ops.reshape(atm_bias, (-1, 1))
            gate_logits_biased = ops.concatenate([
                gate_logits[:, 0:1] + atm_bias_reshaped,
                gate_logits[:, 1:]
            ], axis=-1)
            gate_weights = ops.softmax(gate_logits_biased)
        else:
            gate_weights = self.gating_network(x, training=False)
        
        gate_weights_np = ops.convert_to_numpy(gate_weights)
        
        analysis = {
            'gate_weights': gate_weights_np,
            'expert_activations': gate_weights_np.mean(axis=0),
            'gate_entropy': -ops.convert_to_numpy(ops.sum(gate_weights * ops.log(gate_weights + 1e-8), axis=1)),
            'dominant_expert': gate_weights_np.argmax(axis=1),
        }
        
        # Add ATM-specific analysis if enabled
        if self.atm_specialization:
            m_orig = inputs[1]
            atm_mask = ops.convert_to_numpy(ops.abs(m_orig - 1.0) < 0.1).flatten()
            if atm_mask.sum() > 0:
                analysis['atm_expert_usage'] = gate_weights_np[atm_mask, 0].mean()  # Expert 0 usage for ATM
                analysis['atm_vs_wing_specialization'] = {
                    'atm_points_expert0': gate_weights_np[atm_mask, 0].mean(),
                    'wing_points_expert0': gate_weights_np[~atm_mask, 0].mean() if (~atm_mask).sum() > 0 else 0.0
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

    def build_training_data_from_surfaces(self, Z_latent, Y_surface_flat, strike_tensor, tau_tensor, F_features=None):
        M, K = self.M, self.K
        m_grid, tau_grid = np.meshgrid(strike_tensor, tau_tensor)
        m_flat = m_grid.reshape(-1, 1)
        tau_flat = tau_grid.reshape(-1, 1)

        X_f, X_m, X_tau, y = [], [], [], []
        for i, (z_vec, surf_flat) in enumerate(zip(Z_latent, Y_surface_flat)):
            if self.feature_dim > 0 and F_features is not None:
                zf = ops.convert_to_numpy(ops.concatenate([z_vec, F_features[i]], axis=-1))
            else:
                zf = z_vec
            z_repeat = np.repeat(zf[None, :], M * K, axis=0)
            X_f.append(z_repeat)
            X_m.append(m_flat)
            X_tau.append(tau_flat)
            y.append(surf_flat.reshape(-1, 1))

        return [np.vstack(X_f), np.vstack(X_m), np.vstack(X_tau)], np.vstack(y)


    def _compute_expert_biases(self, m, tau):
        batch_size = ops.shape(m)[0]
        biases = ops.zeros((batch_size, self.num_experts))
        
        # AUTOMATED APPROACH: Temperature annealing (Fedus et al., 2021)
        # Start with strong biases, gradually reduce to let model learn naturally
        if hasattr(self, '_training_step'):
            self._training_step += 1
        else:
            self._training_step = 0
        
        # Auto-calculate annealing schedule based on training configuration
        if not hasattr(self, '_total_training_steps'):
            # Try to get epochs from model's training state
            planned_epochs = getattr(self, 'epochs', None)
            batch_size = getattr(self, 'batch_size', None)
            training_samples = getattr(self, 'training_samples', None)
            
            # Calculate if we have the info, otherwise use simple heuristic
            if all(x is not None for x in [planned_epochs, batch_size, training_samples]):
                steps_per_epoch = training_samples // batch_size
                self._total_training_steps = planned_epochs * steps_per_epoch
            else:
                # Simple heuristic: assume reasonable annealing duration
                self._total_training_steps = 2000  # Most MoE papers use 1000-3000 steps
        
        # Anneal over first 20% of training
        annealing_fraction = getattr(self, 'annealing_fraction', 0.35)
        annealing_steps = int(self._total_training_steps * annealing_fraction)
        
        # Annealing schedule: strong → moderate → gentle
        progress = min(1.0, self._training_step / annealing_steps)
        max_bias_start = getattr(self, 'max_bias_start', 12.0)
        max_bias_end = getattr(self, 'max_bias_end', 4.0)
        moderate_bias_start = getattr(self, 'moderate_bias_start', 4.0)
        moderate_bias_end = getattr(self, 'moderate_bias_end', 1.0)
        
        max_bias = max_bias_end + (max_bias_start - max_bias_end) * (1.0 - progress)
        moderate_bias = moderate_bias_end + (moderate_bias_start - moderate_bias_end) * (1.0 - progress)
        
        if self.maturity_specialization:
            # Aligned with actual maturity grid: [0.083, 0.167, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            tau_bands = [0.3, 0.75, 2, 3.5]  # Group ultra-short, separate medium problem zone, long-term
            bias_strengths = [
            max_bias,              # Ultra-short: 8.0→2.0 (keep high)
            moderate_bias,         # Short-medium: 4.0→1.0 (reduce from +1)  
            moderate_bias + 2.0,   # Problem zone: 5.0→2.0 (increase focus)
            moderate_bias+1.0,         # Long-term: 4.0→1.0 (reduce from 2.5)
            moderate_bias +1.5    # Background: 3.0→0.0 (reduce from 1)
        ] # Stronger bias for 1.5-3.0 range
            
            for i in range(self.maturity_experts):
                if i == 0:  # Ultra-short cluster: 0.083, 0.167, 0.25
                    condition = tau < tau_bands[0]  # τ < 0.3
                    bias_strength = bias_strengths[0]
                elif i == 1:  # Short-medium: 0.5, 0.75 
                    condition = (tau >= tau_bands[0]) & (tau < tau_bands[1])  # 0.3 ≤ τ < 0.75
                    bias_strength = bias_strengths[1]
                elif i == 2:  # Problem zone: 1.0, 1.5 (where gap widens)
                    condition = (tau >= tau_bands[1]) & (tau < tau_bands[2])  # 0.75 ≤ τ < 1.5
                    bias_strength = bias_strengths[2]  # Stronger bias 2.5
                else:  # Long-term: 2.0, 3.0, 4.0, 5.0 (where gap widens more)
                    condition = tau >= tau_bands[2]  # τ ≥ 1.5
                    bias_strength = bias_strengths[3]
                
                maturity_bias = ops.where(condition, bias_strength, 0.0)
                expert_bias = ops.zeros((batch_size, self.num_experts))
                expert_bias = ops.slice_update(expert_bias, (0, i), 
                                            ops.expand_dims(ops.squeeze(maturity_bias, axis=-1), axis=1))
                biases = biases + expert_bias
        
        # Strike specialization with automated calendar arbitrage prevention
        remaining_experts = self.num_experts - self.maturity_experts
        if remaining_experts > 0:
            strike_bands = [0.9, 1.1]  # ITM, ATM, OTM thresholds
            
            for i in range(min(remaining_experts, 3)):
                expert_idx = self.maturity_experts + i
                
                if i == 0:  # ITM expert
                    condition = m < strike_bands[0]
                    bias_strength = max_bias *(1- progress)  # Stronger ITM focus
                elif i == 1:  # ATM expert - AUTOMATED CALENDAR FIX
                    condition = (m >= strike_bands[0]) & (m <= strike_bands[1])
                    
                    # Auto-scaling calendar fix based on training progress
                    calendar_strength = max_bias * 1.5 * (1.0 - progress)  # Scale down over time

                    # Progressive calendar enforcement
                    ultra_short_boost = ops.where(tau < 0.3, calendar_strength, 0.0)
                    short_boost = ops.where((tau >= 0.3) & (tau <0.85), moderate_bias, 0.0)
                    medium_boost = ops.where((tau >= 0.85) & (tau < 2), 2.0, 0.0)
                    long_boost = ops.where(tau >= 2, moderate_bias, 0.0)  # NEW

                    bias_strength = 1.0 + ultra_short_boost + short_boost + medium_boost + long_boost
                else:  # OTM expert
                    condition = m > strike_bands[1]
                    bias_strength = max_bias*2.5*(1- progress)  # Moderate OTM focus
                
                strike_bias = ops.where(condition, bias_strength, 0.0)
                expert_bias = ops.zeros((batch_size, self.num_experts))
                expert_bias = ops.slice_update(expert_bias, (0, expert_idx),
                                            ops.expand_dims(ops.squeeze(strike_bias, axis=-1), axis=1))
                biases = biases + expert_bias
        
        return biases



    # UPDATE get_config method:
    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "M": self.M,
            "K": self.K,
            "taus": self.taus.tolist(),
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
            # NEW:
            "maturity_experts": self.maturity_experts,
            "free_experts": self.free_experts,
            "maturity_specialization": self.maturity_specialization
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)