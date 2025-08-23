import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Add, Attention, BatchNormalization
from keras.optimizers import Adam
# import sequences
from keras.utils import Sequence
from keras import ops

class EncodedLatentSequence(Sequence):
    """
    sequence wrapper for encoding latent sequences from surface and feature tensors
    input: dataset (Sequence), encoder (model)
    output: z_seq (batch, lookback, latent_dim), z_target (batch, latent_dim)
    """
    center_z_seq = False
    
    def __init__(self, dataset, encoder, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.encoder = encoder
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # get batch from dataset
        (surf_seq, feat_seq), surf_target = self.dataset[idx]
        
        # extract dimensions
        B, L, M, K, C = surf_seq.shape  # batch, lookback, maturities, strikes, channels
        D = feat_seq.shape[-1]  # feature dimension
        
        # extract iv channel (last channel)
        surf_seq_iv = surf_seq[..., -1]
        surf_target_iv = surf_target[..., -1]
        
        # flatten sequence for encoder (batch*lookback, surface_dim)
        surf_seq_flat = ops.reshape(surf_seq_iv, (B * L, M * K))
        feat_seq_flat = ops.reshape(feat_seq, (B * L, D))
        
        # concatenate surface and features for encoder input
        encoder_input_seq = ops.concatenate([surf_seq_flat, feat_seq_flat], axis=-1)
        
        # encode sequence
        z_out_seq = self.encoder.predict(encoder_input_seq, verbose=0)
        z_seq = z_out_seq[1] if isinstance(z_out_seq, (list, tuple)) else z_out_seq  # handle vae tuple output
        z_seq = z_seq.reshape(B, L, -1)  # reshape back to sequence format
        
        # optional centering for transformers
        if self.center_z_seq:
            z_seq = z_seq - ops.mean(z_seq, axis=1, keepdims=True)
        
        # prepare target for encoding
        surf_target_flat = ops.reshape(surf_target_iv, (B, M * K))
        feat_target = feat_seq[:, -1, :]  # use last timestep features
        encoder_input_target = ops.concatenate([surf_target_flat, feat_target], axis=-1)
        
        # encode target
        z_out_target = self.encoder.predict(encoder_input_target, verbose=0)
        z_target = z_out_target[1] if isinstance(z_out_target, (list, tuple)) else z_out_target
        
        return z_seq, z_target
    
def center_z_sequences(X_seq):
    """
    center latent sequences by subtracting mean
    input: X_seq (array)
    output: centered array
    """
    mean = np.mean(X_seq, axis=(0, 1), keepdims=True)
    return X_seq - mean

def build_z_sequences(Z, lookback):
    """
    build input/output sequences for lstm from latent Z
    input: Z (array), lookback (int)
    output: centered X_seq, y_seq
    """
    X_seq, y_seq = [], []
    for t in range(lookback, len(Z)):
        X_seq.append(Z[t - lookback: t])
        y_seq.append(Z[t])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    return center_z_sequences(X_seq), y_seq

def build_lstm_forecaster(lookback, latent_dim, lr=None):
    """
    build lstm forecaster model for latent sequence prediction
    input: lookback (int), latent_dim (int), lr (float)
    output: compiled keras model
    """
    lstm_input = Input(shape=(lookback, latent_dim), name="latent_sequence_input")
    x = LSTM(64, return_sequences=True, activation="tanh")(lstm_input)
    x = Dropout(0.2)(x)
    x = LSTM(32, activation="tanh")(x)
    x = Dropout(0.1)(x)
    z_base = lstm_input[:, -1, :]
    z_delta = Dense(latent_dim, name="delta_z")(x)
    z_pred = Add(name="residual_output")([z_base, z_delta])
    model = Model(lstm_input, z_pred, name="lstm_forecaster")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

def build_lstm_with_attention(lookback, latent_dim, lr=None):
    """
    build lstm model with attention for latent sequence prediction
    input: lookback (int), latent_dim (int), lr (float)
    output: compiled keras model
    """
    lstm_input = Input(shape=(lookback, latent_dim), name="latent_sequence_input")
    x = LSTM(64, return_sequences=True, activation="tanh")(lstm_input)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True, activation="tanh")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    attention_output = Attention()([x, x])
    attention_output = BatchNormalization()(attention_output)
    z_base = lstm_input[:, -1, :]
    z_delta = Dense(latent_dim, name="delta_z")(attention_output[:, -1, :])
    z_pred = Add(name="residual_output")([z_base, z_delta])
    model = Model(lstm_input, z_pred, name="lstm_with_attention")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

def forecast_latent_sequence(model, X_seq, test_idx):
    """
    forecast z(t+1) from tail of latent sequence x_seq, aligned with test_idx
    input: model, x_seq (array), test_idx (array)
    output: z_pred (array), n_pred (int)
    """
    N_pred = min(len(test_idx), X_seq.shape[0])
    Z_pred = model.predict(X_seq[-N_pred:])
    return Z_pred, N_pred

def build_lstm_with_attention_augmented(lookback, input_dim, latent_dim, lr=None):
    """
    build lstm with attention for augmented input (z, f)
    input: lookback (int), input_dim (int), latent_dim (int), lr (float)
    output: compiled keras model
    """
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Attention
    from keras.optimizers import Adam
    lstm_input = Input(shape=(lookback, input_dim), name="latent_sequence_input")
    x = LSTM(64, return_sequences=True, activation="tanh")(lstm_input)
    x = LSTM(64, return_sequences=True, activation="tanh")(x)
    x = LSTM(32, return_sequences=True, activation="tanh")(x)
    attention_output = Attention()([x, x])
    x_final = attention_output[:, -1, :]
    z_pred = Dense(latent_dim, name="z_output")(x_final)
    model = Model(lstm_input, z_pred, name="lstm_with_attention_augmented")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def build_z_sequences_augmented(Z_aug, lookback, latent_dim):
    """
    build input/output sequences for lstm from augmented latent Z
    input: Z_aug (array), lookback (int), latent_dim (int)
    output: X_seq, y_seq (arrays)
    """
    X_seq, y_seq = [], []
    for t in range(lookback, len(Z_aug)):
        X_seq.append(Z_aug[t - lookback:t])
        y_seq.append(Z_aug[t][:latent_dim])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def build_lstm_forecaster_augmented(lookback, input_dim, latent_dim, lr=None):
    """
    build lstm forecaster for augmented input (z, f)
    input: lookback (int), input_dim (int), latent_dim (int), lr (float)
    output: compiled keras model
    """
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense
    from keras.optimizers import Adam
    lstm_input = Input(shape=(lookback, input_dim), name="latent_sequence_input")
    x = LSTM(64, return_sequences=True, activation="tanh")(lstm_input)
    x = LSTM(64, return_sequences=True, activation="tanh")(x)
    x = LSTM(32, activation="tanh")(x)
    z_pred = Dense(latent_dim, name="z_output")(x)
    model = Model(lstm_input, z_pred, name="lstm_forecaster_augmented")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

