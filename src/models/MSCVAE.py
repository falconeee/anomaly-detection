import pandas as pd
import numpy as np
import math
import random
import os
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset

# Helper Classes
class AttributeMatrixGenerator:
    """
    Transforms raw multivariate time-series into spatial-temporal correlation matrices
    and target values for the Hybrid MSCVAE model.
    """
    def __init__(self, window_sizes=(10, 30, 60), step=10):
        # Multi-scale temporal windows. Each scale s produces an N x N attribute matrix
        # M_t^(s) = X_{t-s:t} X_{t-s:t}^T / s, stacked along the channel dimension so the CNN
        # sees short- and long-range correlation structure simultaneously. This realizes the
        # "Multi-Scale" of MSCVAE/MSCRED instead of a single fixed window. Scales are
        # de-duplicated and sorted ascending.
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes]
        self.window_sizes = sorted({int(w) for w in window_sizes})
        # Effective window used for time-alignment / minimum history: the LARGEST scale,
        # since a sample only exists once every scale has a full window of history behind it.
        self.w = max(self.window_sizes)
        self.step = step     # Sliding window stride (step=w means no overlap)
        self.mean = None
        self.std = None
        # Physical (raw, un-normalized) lower bound per variable. Populated by fit_scaler.
        # Used to clip denormalized reconstructions back into a physically plausible range.
        self.min_physical_vals = {}

    def fit_scaler(self, train_dataframes):
        """
        Calculates the mean and standard deviation from the training data.
        These statistics are used to normalize the data (Z-score) before generating matrices.
        """
        if isinstance(train_dataframes, pd.DataFrame):
            train_dataframes = [train_dataframes]

        # Concat to calculate global statistics (mean, std)
        # Z-score normalization is crucial so features with larger magnitudes do not dominate the dot product calculations.
        full_train_df = pd.concat(train_dataframes, ignore_index=True)
        self.mean = full_train_df.mean()
        self.std = full_train_df.std() + 1e-6 # Added epsilon to prevent division by zero
        # Store the physical minimum observed per variable in the (normal) training data.
        self.min_physical_vals = full_train_df.min().to_dict()

    def generate(self, df):
        if self.mean is None:
            raise ValueError("Execute .fit_scaler() first!")

        # Scaling data to mean=0, std=1
        data = (df - self.mean) / self.std
        # NaNs become 0 (which is the mean after scaling), preserving matrix stability
        values = np.nan_to_num(data.values)

        matrices = []
        target_values = []

        if len(values) < self.w:
            # Return empty tuple if df is too small (i.e. smaller than the largest window)
            return torch.empty(0), torch.empty(0)

        values_t = torch.tensor(values, dtype=torch.float32)

        # Sliding window extraction. The window END is driven by the LARGEST scale so every
        # scale has full history; smaller scales simply read their own (shorter) tail.
        for t in range(self.w, len(values), self.step):
            scale_mats = []
            for w_s in self.window_sizes:
                x_segment = values_t[t - w_s:t]                  # (w_s, n_features)

                # Matrix (Eq. 1 from paper) - Spatial-temporal inner product at scale w_s.
                # Captures correlations between all pairs of sensors within the window.
                # Kernel Trick can be applied here but the model already has a non-linear
                # decoder and the anomaly detection gets less sensitive.
                x_t = x_segment.T                                # (n_features, w_s)
                m_t = torch.matmul(x_t, x_t.T) / w_s             # (n_features, n_features)
                scale_mats.append(m_t)

            # Stack scales on the channel dimension -> (n_scales, n_features, n_features)
            matrices.append(torch.stack(scale_mats, dim=0))

            # Exact raw values of the last timestamp in the window (shared across scales).
            # Used as the ground truth for the Hybrid MLP val_decoder.
            target_values.append(values_t[t - 1])

        if not matrices:
            return torch.empty(0), torch.empty(0)

        # Tuple: (Tensor of Matrices, Tensor of Values)
        # The channel dimension (B, n_scales, N, N) carries the scales for Conv2d layers.
        return torch.stack(matrices), torch.stack(target_values)


class SequenceMatrixDataset(Dataset):
    """
    Wraps the flat list of attribute matrices into temporally-ordered SEQUENCES.

    For each target window i, the item is the stack of the last `seq_len` matrices
    (i-seq_len+1 ... i), left-padded by repeating the earliest available matrix.
    The target value is the raw vector of the *current* (last) window.

    Why this matters: the temporal context now travels INSIDE each item, so it no
    longer depends on batch boundaries or on the DataLoader shuffle order. This is
    what makes the anomaly scores reproducible (independent of batch_size) and lets
    training shuffle batches without feeding the transformer unrelated windows.
    """
    def __init__(self, matrices, values, seq_len):
        # matrices: (num, 1, N, N) ; values: (num, N)
        self.matrices = matrices
        self.values = values
        self.seq_len = seq_len

    def __len__(self):
        return self.matrices.size(0)

    def __getitem__(self, i):
        idxs = [max(0, j) for j in range(i - self.seq_len + 1, i + 1)]
        seq = self.matrices[idxs]          # (seq_len, 1, N, N)
        return seq, self.values[i]         # value of the current window


class SpatialTemporalTransformer(nn.Module):
    """
    Spatial-Temporal Transformer to replace ConvLSTM.
    Processes the sequence of spatial correlation matrices in parallel across the temporal dimension.
    """
    def __init__(self, channels=64, dim_feedforward=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, channels)
        )

    def forward(self, sequence_tensor):
        B, T, C, H, W = sequence_tensor.size()
        x = sequence_tensor.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        # Inject temporal order: self-attention is permutation-invariant, so without a
        # positional signal this module collapses into an order-agnostic pooling over the
        # window sequence. A sinusoidal encoding restores temporal ordering, letting the
        # transformer actually model the evolution/inertia of the system state.
        x = x + self._temporal_positional_encoding(T, C, x.device, x.dtype).unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        out = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return out[:, -1, :, :, :]

    @staticmethod
    def _temporal_positional_encoding(T, C, device, dtype):
        """
        Deterministic sinusoidal positional encoding of shape (T, C) over the time axis.
        Parameter-free (no learned weights, so it adds no reproducibility risk) and works
        for any sequence length T, including the degenerate T=1 case.
        """
        pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)      # (T, 1)
        idx = torch.arange(0, C, 2, device=device, dtype=dtype)             # (C/2,)
        div = torch.exp(-math.log(10000.0) * idx / C)
        pe = torch.zeros(T, C, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe


class MSCVAE_Hybrid(nn.Module):
    """
    Multivariate Spatial-Temporal Convolutional Variational Autoencoder (Hybrid).
    Architecture:
    1. CNN Encoder: Compresses spatial correlation matrices into a latent space.
    2. VAE Core (mu, logvar): Learns a continuous normal distribution of the normal system state.
    3. Spatial-Temporal Transformer: Captures the temporal evolution (inertia) of the
       compressed states over an EXPLICIT sequence of consecutive windows.
    4. Dual Decoder (Hybrid mechanism):
       - Route 1 (MLP): Reconstructs exact raw values (sensitive to extreme peaks).
       - Route 2 (CNN): Reconstructs correlation matrices (sensitive to relationship breaks).

    Input contract:
       forward(x) expects x of shape (B, T, n_scales, N, N) -- a batch of temporal sequences,
       where the channel dimension carries the multi-scale attribute matrices.
       A 4D tensor (B, n_scales, N, N) is accepted as a degenerate sequence of length 1.
       The matrix that is reconstructed / scored is always the LAST timestep (x[:, -1]).
    """
    def __init__(self, n_features, n_scales=1):
        super(MSCVAE_Hybrid, self).__init__()
        self.n_features = n_features
        # Number of multi-scale attribute matrices stacked as input/output channels.
        self.n_scales = n_scales
        # Latent dimension scaled by the square root of features to prevent over-compression
        latent_dim = round(math.sqrt(n_features))

        # Encoder: Extracts hierarchical spatial features from the (multi-scale) matrices
        self.enc1 = nn.Conv2d(n_scales, 16, 3, 2, 1)
        self.enc2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.enc3 = nn.Conv2d(32, 64, 3, 2, 1)

        # Dummy pass: Automatically calculates the flattened dimension size after convolutions.
        # This makes the architecture dynamic and adaptable to any number of sensors (n_features).
        with torch.no_grad():
            dummy = torch.zeros(1, n_scales, n_features, n_features)
            out = self.enc3(self.enc2(self.enc1(dummy)))
            self.flatten_dim = out.view(1, -1).size(1)
            self.spatial_shape = out.shape[1:]

        # Latent Space (VAE Core)
        # Projects the flattened CNN features into probabilistic mean and variance vectors
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder Matrix: Projects the latent vector back to the spatial shape required by ConvTranspose2d
        self.fc_decode_mat = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder of Values (Hybrid)
        # MLP network designed to reconstruct raw sensor values by combining the static
        # latent representation (z) with temporal memory (h_att).
        val_input_dim = latent_dim + self.flatten_dim
        self.val_decoder = nn.Sequential(
            nn.Linear(val_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, n_features)
        )

        # Temporal Modeling
        self.transformer = SpatialTemporalTransformer(channels=64)

        # Convolutional Decoder: Upsamples the combined (Z + Temporal) features back to the
        # original N x N matrices, one output channel per scale.
        self.dec3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1)
        self.dec1 = nn.ConvTranspose2d(16, n_scales, 3, 2, 1, output_padding=1)

        # Homoscedastic uncertainty parameters (learned task balancing, Kendall et al.)
        self.log_var_mat = nn.Parameter(torch.zeros(1))
        self.log_var_val = nn.Parameter(torch.zeros(1))

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: Allows backpropagation through the stochastic sampling process.
        During training, it adds Gaussian noise (std) to the mean (mu).
        During evaluation (eval), it acts deterministically returning only the mean, stabilizing anomaly scores.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        else:
            return mu

    def forward(self, x):
        # Accept a degenerate 4D input as a length-1 sequence.
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T = x.size(0), x.size(1)
        C_in = x.size(2)  # number of scale channels
        N = x.size(-1)

        # Encode every timestep in the sequence in a single batched pass.
        x_flat = x.reshape(B * T, C_in, N, N)
        e3 = F.relu(self.enc3(F.relu(self.enc2(F.relu(self.enc1(x_flat))))))
        C, h, w = e3.size(1), e3.size(2), e3.size(3)
        e3_seq = e3.view(B, T, C, h, w)

        # VAE core operates on the CURRENT (last) timestep.
        e3_cur = e3_seq[:, -1]
        flat = e3_cur.reshape(B, -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        # Temporal Modeling over the explicit, ordered sequence.
        h_att = self.transformer(e3_seq)  # (B, C, h, w)

        # Route 1: Raw Values (Hybrid Decoding)
        flat_h_att = h_att.reshape(B, -1)
        z_temporal = torch.cat([z, flat_h_att], dim=1)
        recon_values = self.val_decoder(z_temporal)

        # Route 2: Matrices (Standard Decoding)
        z_dec = self.fc_decode_mat(z).view(B, *self.spatial_shape)
        combined = z_dec + h_att

        d3 = F.relu(self.dec3(combined))
        d2 = F.relu(self.dec2(d3))
        recon_matrix = self.dec1(d2)

        # Bilinear interpolation ensures output matches the exact N x N input dimensions.
        if recon_matrix.shape[2:] != (N, N):
            recon_matrix = F.interpolate(recon_matrix, size=(N, N), mode='bilinear', align_corners=False)

        return recon_matrix, recon_values, mu, logvar

    def loss_function(self, recon_matrix, x_matrix, recon_values, x_values, mu, logvar, beta=0.6):
        n_features = x_matrix.shape[2]
        # Per-sample element count of the matrix target now spans all scale channels.
        n_elements_matrix = x_matrix.shape[1] * x_matrix.shape[2] * x_matrix.shape[3]

        mse_mat_mean = F.mse_loss(recon_matrix, x_matrix, reduction='mean')
        mse_val_mean = F.mse_loss(recon_values, x_values, reduction='mean')

        # Re-express per-element means as per-sample sums over each target's own size:
        #   matrix target has N*N elements -> multiply by N*N
        #   value  target has N   elements -> multiply by N
        # This keeps both reconstruction terms on a comparable, geometry-aware scale.
        MSE_Mat_scaled = mse_mat_mean * n_elements_matrix
        MSE_Val_scaled = mse_val_mean * n_features

        precision_mat = torch.exp(-self.log_var_mat)
        loss_mat = precision_mat * MSE_Mat_scaled + self.log_var_mat

        precision_val = torch.exp(-self.log_var_val)
        loss_val = precision_val * MSE_Val_scaled + self.log_var_val

        # KL divergence per sample (mean over batch), so the term is independent of
        # batch_size and beta stays portable across configurations.
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        total_loss = loss_mat + loss_val + (beta * KLD)
        return total_loss

class SPOT:
    """
    Streaming Peaks-Over-Threshold (SPOT) Algorithm.
    Purpose: Dynamically calculates anomaly thresholds based on Extreme Value Theory (EVT).
    Instead of assuming a normal (Gaussian) distribution of errors, SPOT models the
    'tail' of the distribution (the extreme reconstruction errors) using a Generalized
    Pareto Distribution (GPD). This allows for robust, mathematically sound thresholding
    that adapts to non-linear and heavy-tailed error distributions typical in VAEs.
    """
    def __init__(self, q=1e-4):
        # q (proba): The desired probability of false alarms. A lower 'q' means a
        # stricter threshold, resulting in fewer anomalies being flagged.
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, init_data, data):
        """
        Loads the initial calibration data (from the training phase) and the
        data to be monitored. Handles multiple data types (list, numpy, pandas).
        """
        if isinstance(data, list): self.data = np.array(data)
        elif isinstance(data, np.ndarray): self.data = data
        elif isinstance(data, pd.Series): self.data = data.values
        else: return
        if isinstance(init_data, list): self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray): self.init_data = init_data
        elif isinstance(init_data, pd.Series): self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (init_data < 1) and (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else: return

    def add(self, data):
        """Appends new streaming data to the existing data array."""
        if isinstance(data, list): data = np.array(data)
        elif isinstance(data, np.ndarray): data = data
        elif isinstance(data, pd.Series): data = data.values
        else: return
        self.data = np.append(self.data, data)

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Calibration Phase: Establishes the base parameters using normal data.
        It sorts the initial data and sets a basic threshold (e.g., the 98th percentile).
        Values above this are considered 'peaks' and are used to fit the GPD.
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            self.level = 1 - level
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]

        # 'peaks' are the extreme reconstruction errors that exceed the initial threshold
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        # Fits the Generalized Pareto Distribution to the peaks
        g, s, l = self._grimshaw()
        # Calculates the final strict threshold based on the desired false alarm probability (q)
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """Helper function for Grimshaw's trick: Finds roots of the derivative equation."""
        from scipy.optimize import minimize
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            if step == 0: bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))
        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """Calculates the log-likelihood of the GPD given shape (gamma) and scale (sigma) parameters."""
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + math.log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Grimshaw's Trick: An efficient algorithm to find the Maximum Likelihood Estimation (MLE)
        for the parameters of the Generalized Pareto Distribution (gamma and sigma)
        based on the observed peaks.
        """
        def u(s): return 1 + np.log(s).mean()
        def v(s): return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            us = u(s); vs = v(s)
            return us * vs - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s); vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon: epsilon = abs(a) / n_points
        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')
        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')
        zeros = np.concatenate((left_zeros, right_zeros))

        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Computes the final anomaly threshold (extreme quantile) using the fitted GPD parameters
        and the predefined false alarm probability (q).
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0: return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: return self.init_threshold - sigma * math.log(r)

    def run(self, with_alarm=True, dynamic=True):
        """
        Streaming Inference Phase.
        Iterates over the test data. If 'dynamic=True', the GPD parameters and the threshold
        are updated on-the-fly whenever a new peak (that is not an anomaly) is observed.
        Returns a dictionary containing the dynamic thresholds and the indices of anomalies.
        """
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you should initialize before running again')
            return {}

        th = []
        alarm = []

        for i in range(self.data.size):
            if not dynamic:
                # Static evaluation: Threshold never updates.
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # Dynamic evaluation: Updates the GPD distribution with new normal peaks.
                if self.data[i] > self.extreme_quantile:
                    # Data point exceeds the extreme quantile: It's an Anomaly.
                    if with_alarm: alarm.append(i)
                    else:
                        # If alarms are off, treat the anomaly as a new peak and update distribution.
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)
                elif self.data[i] > self.init_threshold:
                    # Data point is a peak but NOT an anomaly: Update the GPD to learn the new normal.
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    # Normal data point below initial threshold.
                    self.n += 1

            th.append(self.extreme_quantile)

        return {'thresholds': th, 'alarms': alarm}


class MSCVAE:
    """
    Main wrapper class for the MSCVAE anomaly detection pipeline.
    Purpose: Acts as the high-level API orchestrating the entire lifecycle:
    data preprocessing (matrix generation), model training, dynamic threshold
    calculation (SPOT), and root cause analysis (contribution).
    """
    def __init__(self, n_features=None, window_sizes=None, window_size=None, stride=1,
                 seq_len=5, device=None, seed=42):
        self.seed = seed
        # Enforce reproducibility right at initialization
        self.set_deterministic(self.seed)

        self.n_features = n_features
        # Multi-scale windows. Default is the paper-inspired {10, 30, 60}. Back-compat: a
        # single `window_size` (int) still produces a single-scale model; `window_sizes`
        # (list/tuple) takes precedence when provided.
        if window_sizes is None:
            window_sizes = (10, 30, 60) if window_size is None else window_size
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes]
        self.window_sizes = sorted({int(w) for w in window_sizes})
        # Largest scale doubles as the reference window for time-alignment.
        self.window_size = max(self.window_sizes)
        self.stride = stride
        # Number of consecutive windows fed to the temporal transformer per sample.
        self.seq_len = max(1, int(seq_len))

        # Root-cause attribution weights. The matrix DIAGONAL (a variable's own variance
        # reconstruction error) is a clean self-signal; the OFF-DIAGONAL residual is shared
        # between correlated pairs, so a single anomalous variable smears error onto its
        # partners. We down-weight the relational channel to avoid blaming innocents.
        self.rca_self_weight = 1.0
        self.rca_rel_weight = 0.5

        # Hardware selection: automatically defaults to GPU if available for faster tensor operations
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = None
        # Instantiates the generator that will transform flat series into spatial-temporal inputs
        self.generator = AttributeMatrixGenerator(window_sizes=self.window_sizes, step=self.stride)

        self.threshold = None
        # Gain acts as a manual sensitivity tuner applied on top of the SPOT statistical threshold
        self.gain = 1.0

        # Per-variable baseline reconstruction-error statistics (median + MAD), calibrated
        # on the NORMAL training data. These turn raw errors into deviations from each
        # variable's own normal behavior, which is the basis for honest root-cause analysis.
        # The matrix error is split into the DIAGONAL (self/variance) and OFF-DIAGONAL
        # (relational) channels so attribution can lean on the clean self-signal.
        self.mat_diag_med_ = None
        self.mat_diag_mad_ = None
        self.mat_off_med_ = None
        self.mat_off_mad_ = None
        self.val_err_med_ = None
        self.val_err_mad_ = None

    def set_deterministic(self, seed=42):
        """
        Fixes random seeds across all underlying libraries (Python, NumPy, PyTorch).
        Deep learning involves stochastic processes (weight initialization, batch shuffling).
        Locking the seed ensures that experiments, debugging, and anomaly scores are 100%
        reproducible across different executions on the same machine.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # Ensures reproducibility in multi-GPU setups

        # Forces CuDNN to use deterministic convolution algorithms.
        # Prevents slight precision variations caused by underlying hardware optimizations.
        torch.backends.cudnn.deterministic = True
        # Disables auto-tuner that searches for the fastest convolution algorithm,
        # prioritizing consistency over maximum speed.
        torch.backends.cudnn.benchmark = False

    def _seq_loader(self, matrices, values=None, batch_size=128, shuffle=False):
        """
        Builds a DataLoader over temporal SEQUENCES (see SequenceMatrixDataset).
        For score-only paths (no targets) a dummy value tensor is supplied.
        """
        if values is None:
            values = torch.zeros(matrices.size(0), self.n_features)
        ds = SequenceMatrixDataset(matrices, values, self.seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def fit(self, train_data, epochs=50, batch_size=128, lr=1e-3, gain=1.0, verbose=True):
        """
        Main training orchestration pipeline.
        Handles data preprocessing, model weight optimization, and statistical threshold calibration.
        """
        self.gain = gain
        # Standardize input to a list of DataFrames to support training on multiple disjoint periods
        if isinstance(train_data, pd.DataFrame):
            train_data = [train_data]

        # Fit scaler
        if verbose: print("Fitting scaler...")
        # Fits Z-score scaler globally on normal data to ensure stable inner products later
        self.generator.fit_scaler(train_data)

        # If n_features was not set, infer it dynamically from the dataset
        if self.n_features is None:
            self.n_features = train_data[0].shape[1]

        # Prepare data
        if verbose: print("Generating training data...")
        train_matrices = []
        train_values = []
        # One sequence dataset PER period, so temporal sequences never cross the
        # boundary between two disjoint training periods.
        seq_datasets = []

        for df in train_data:
            # Transforms flat multivariate series into spatial-temporal matrices and target values
            t_mat, t_val = self.generator.generate(df)
            if t_mat.nelement() > 0:
                train_matrices.append(t_mat)
                train_values.append(t_val)
                seq_datasets.append(SequenceMatrixDataset(t_mat, t_val, self.seq_len))

        if not seq_datasets:
             raise ValueError("No training data generated. Check window_size and data length!")

        final_train_matrix = torch.cat(train_matrices, dim=0)
        final_train_values = torch.cat(train_values, dim=0)

        # Shuffling batches is now safe: each item already carries its own temporal
        # context, so shuffling no longer feeds the transformer unrelated windows.
        train_loader = DataLoader(
            ConcatDataset(seq_datasets),
            batch_size=batch_size,
            shuffle=True
        )

        # Initialize Model
        self.model = MSCVAE_Hybrid(
            n_features=self.n_features, n_scales=len(self.generator.window_sizes)
        ).to(self.device)
        # Adam optimizer is used due to its adaptive learning rate, which is highly
        # effective for training the distinct components of a Hybrid VAE simultaneously.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Training Phase
        self.model.train()
        if verbose: print(f"Starting training on {self.device} for {epochs} epochs...")

        def calculate_beta_annealing(epoch, max_epochs):
            cycle_length = max(1, max_epochs // 4)
            return min(1.0, (epoch % cycle_length) / (cycle_length * 0.5))

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            beta_val = calculate_beta_annealing(epoch, epochs)

            for batch_seq, batch_values in train_loader:
                x_seq = batch_seq.to(self.device)          # (B, T, 1, N, N)
                x_val = batch_values.to(self.device)

                optimizer.zero_grad()
                recon_mat, recon_val, mu, logvar = self.model(x_seq)
                # The reconstruction target is always the current (last) window.
                target_mat = x_seq[:, -1]

                loss = self.model.loss_function(recon_mat, target_mat, recon_val, x_val, mu, logvar, beta=beta_val)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()
            epoch_duration = time.time() - epoch_start_time

            if verbose and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0):
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader.dataset):.4f} | Time: {epoch_duration:.2f}s")

        # Post-Training: Threshold Calibration (POT)
        if verbose: print("Calculating threshold...")
        # Evaluates the model on the normal training data to establish the baseline error
        # distribution. Uses the COMBINED (matrix + value) score, so SPOT calibrates the
        # threshold on exactly the quantity predict() will compare against.
        train_scores = self._get_anomaly_scores(final_train_matrix, final_train_values)

        # Applies Extreme Value Theory (SPOT algorithm) to find the mathematical upper limit
        # of normal reconstruction errors, providing a strict, unsupervised alarm threshold.
        self.threshold = self._pot_eval(train_scores)

        if verbose:
            print(f"Base Threshold (POT): {self.threshold:.6f}")
            print(f"Gain: {self.gain}")
            print(f"Final Threshold: {self.threshold * self.gain:.6f}")

        # Gain acts as a manual sensitivity multiplier (e.g., gain=1.2 makes alarms 20% less sensitive)
        self.threshold = self.threshold * self.gain

        # Calibrate the per-variable error baseline used by the contribution (RCA) routines.
        if verbose: print("Calibrating per-variable error baseline...")
        self._calibrate_variable_baseline(final_train_matrix, final_train_values)

    def _task_precisions(self):
        """
        Reads the model's learned homoscedastic precisions (exp(-log_var)) for the matrix
        and value reconstruction tasks. These are the model's OWN learned relative weights,
        so reusing them to combine the two error channels keeps the anomaly score on the
        same geometry as the training objective. Returns plain floats (deterministic).
        """
        with torch.no_grad():
            p_mat = float(torch.exp(-self.model.log_var_mat).item())
            p_val = float(torch.exp(-self.model.log_var_val).item())
        return p_mat, p_val

    def _get_anomaly_scores(self, data_tensor, values=None, batch_size=128):
        """
        Calculates the anomaly score (phi) for each window by COMBINING both reconstruction
        routes:
          - the spatial-temporal correlation matrix (sensitive to relationship breaks);
          - the raw-value reconstruction (sensitive to point/contextual peaks that barely
            move the correlation matrix after z-scoring and the 1/w averaging).
        The two channels are weighted by the model's learned homoscedastic precisions, so the
        value route -- trained but historically unused at scoring time -- now contributes to
        detection. When `values` is None the score degrades gracefully to matrix-only.

        Reproducible by construction: temporal context lives inside each sequence item, so
        the score does not depend on batch_size.
        """
        self.model.eval()
        p_mat, p_val = self._task_precisions()
        use_val = values is not None
        scores = []

        loader = self._seq_loader(data_tensor, values, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch_seq, batch_val in loader:
                x = batch_seq.to(self.device).float()      # (B, T, n_scales, N, N)
                recon_mat, recon_val, _, _ = self.model(x)
                target = x[:, -1]                          # (B, n_scales, N, N)

                # Per-window sum of squared error for each route.
                mat_sumsq = (recon_mat - target).pow(2).sum(dim=(1, 2, 3))
                if use_val:
                    v = batch_val.to(self.device).float()
                    val_sumsq = (recon_val - v).pow(2).sum(dim=1)
                    phi = p_mat * mat_sumsq + p_val * val_sumsq
                else:
                    phi = p_mat * mat_sumsq

                scores.extend(phi.cpu().numpy())

        return np.array(scores)

    def _get_variable_errors(self, matrices, values, batch_size=128, return_recon=False):
        """
        Per-WINDOW, per-VARIABLE reconstruction error, decomposed into three channels.

        The N x N matrix residual is summed over the scale channels, then split into a self
        and a relational part instead of being collapsed into a single number. This is what
        lets root-cause analysis avoid spreading blame: when variable i is anomalous, EVERY
        pair (i, j) breaks, so a naive row+col sum implicates every correlated partner j.
        The diagonal does not have this problem.

        Returns
        -------
        mat_diag : ndarray (T, N)
            Matrix DIAGONAL error err[i, i] (summed over scales) -- variable i's own
            variance/energy reconstruction error. The CLEAN self-signal.
        mat_off : ndarray (T, N)
            Matrix OFF-DIAGONAL error (row + column, diagonal removed; summed over scales)
            -- the relational error shared between i and its partners; weaker, contextual
            evidence. Note Sum_i mat_off[t, i] = 2 * (off-diagonal total).
        val_err : ndarray (T, N)
            Squared error of the raw-value reconstruction per variable.
        recon_vals : ndarray (T, N), optional
            The (still z-scored) reconstructed values, returned when return_recon=True.

        The full per-window matrix error used by the detector equals
        mat_diag.sum(1) + mat_off.sum(1) / 2.
        """
        self.model.eval()
        loader = self._seq_loader(matrices, values, batch_size=batch_size, shuffle=False)

        diag_list, off_list, val_list, recon_list = [], [], [], []
        with torch.no_grad():
            for batch_seq, batch_val in loader:
                x = batch_seq.to(self.device).float()          # (B, T, n_scales, N, N)
                recon_mat, recon_val, _, _ = self.model(x)
                target = x[:, -1]                              # (B, n_scales, N, N)

                err = (target - recon_mat).pow(2).sum(dim=1)   # sum scales -> (B, N, N)
                diag = torch.diagonal(err, dim1=1, dim2=2)     # (B, N) self error
                row = err.sum(dim=2)                           # (B, N) includes diagonal
                col = err.sum(dim=1)                           # (B, N) includes diagonal
                off = (row - diag) + (col - diag)              # (B, N) off-diagonal only
                diag_list.append(diag.cpu().numpy())
                off_list.append(off.cpu().numpy())

                v = batch_val.to(self.device).float()
                val_list.append((v - recon_val).pow(2).cpu().numpy())
                if return_recon:
                    recon_list.append(recon_val.cpu().numpy())

        mat_diag = np.concatenate(diag_list, axis=0)
        mat_off = np.concatenate(off_list, axis=0)
        val_err = np.concatenate(val_list, axis=0)
        if return_recon:
            return mat_diag, mat_off, val_err, np.concatenate(recon_list, axis=0)
        return mat_diag, mat_off, val_err

    def _calibrate_variable_baseline(self, matrices, values, batch_size=128):
        """
        Stores the median and (scaled) MAD of each variable's reconstruction error over the
        NORMAL training data, separately for the matrix diagonal (self), matrix off-diagonal
        (relational) and raw-value channels. Robust statistics are used so a few extreme
        windows do not inflate the baseline. These define what 'normal error' looks like per
        variable and per channel.
        """
        mat_diag, mat_off, val_err = self._get_variable_errors(matrices, values, batch_size=batch_size)
        k = 1.4826  # makes MAD comparable to a Gaussian standard deviation

        def _med_mad(a):
            med = np.median(a, axis=0)
            mad = np.median(np.abs(a - med), axis=0) * k + 1e-9
            return med, mad

        self.mat_diag_med_, self.mat_diag_mad_ = _med_mad(mat_diag)
        self.mat_off_med_, self.mat_off_mad_ = _med_mad(mat_off)
        self.val_err_med_, self.val_err_mad_ = _med_mad(val_err)

    def _pot_eval(self, init_score, q=1e-4, level=0.02):
        """
        Wrapper for the SPOT (Peaks-Over-Threshold) algorithm initialization.
        Finds the exact mathematical threshold (extreme_quantile) above which
        a reconstruction error is considered a true anomaly, rather than just noise.
        """
        lms = level
        while True:
            try:
                s = SPOT(q)
                s.fit(init_score, init_score)
                # Attempts to fit the Pareto distribution. If the data is too smooth or
                # lacks distinct peaks, the Scipy optimization might fail (raise Exception).
                s.initialize(level=lms, min_extrema=False, verbose=False)
            except Exception:
                # Fallback mechanism: Gradually lowers the definition of what constitutes a 'peak'
                # until the optimization algorithm converges successfully.
                lms = lms * 0.999
            else:
                break
        return s.extreme_quantile

    def predict(self, df_test, timestamps=None, batch_size=128):
        """
        Inference pipeline for new unseen data.
        Returns a dictionary mapping timestamps to their respective anomaly scores (phi).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first!")

        self.model.eval()

        # Transform raw test data into correlation matrices (and the matching raw values).
        try:
            tensor_matrices, tensor_values = self.generator.generate(df_test)
        except ValueError as e:
            print(f"Generation error: {e}")
            return {}

        if tensor_matrices.nelement() == 0:
            print(f"Test dataframe too small for window {self.generator.w}.")
            return {}

        # Combined anomaly score (matrix + value routes) for each window.
        all_scores = self._get_anomaly_scores(tensor_matrices, tensor_values, batch_size=batch_size)

        # Time Alignment
        # The generator uses a sliding window (w) and a step size (s).
        # This means the score calculated for matrix M_t actually represents the state
        # of the system at the *end* of that window. We must map the score back to the
        # correct original timestamp to avoid time-shift errors in production.
        w = self.generator.w
        s = self.generator.step

        if timestamps is not None:
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)

            # 'valid_indices' mimics the loop in the generator to find which timestamps
            # correspond to the end of each generated matrix.
            valid_indices = range(w, len(df_test) + 1, s)

            # Truncate to the smallest length to prevent IndexError in case of minor dimension mismatches
            min_len = min(len(all_scores), len(valid_indices))
            final_scores = all_scores[:min_len]
            final_indices = list(valid_indices)[:min_len]

            final_timestamps = []
            for idx in final_indices:
                # Basic protection for index out of bounds
                if idx < len(ts_values):
                     final_timestamps.append(ts_values[idx])
                elif idx == len(ts_values):
                    # If index hits exactly the length, grab the last available timestamp
                    final_timestamps.append(ts_values[-1])

            final_scores = final_scores[:len(final_timestamps)]

            return {
                'timestamp': final_timestamps,
                'phi': final_scores
            }
        else:
            return {
                'timestamp': np.arange(len(all_scores)), # Dummy timestamps if none provided
                'phi': all_scores
            }

    def _select_anomaly_windows(self, phi, anomaly_windows=None):
        """
        Returns a boolean mask over the generated windows selecting those that belong to
        the detected anomaly. Priority:
          1. explicit `anomaly_windows` indices, if provided;
          2. windows whose phi exceeds the calibrated threshold;
          3. fallback to the single worst window, so RCA always has something to explain.
        """
        mask = np.zeros(len(phi), dtype=bool)
        if anomaly_windows is not None:
            idx = np.atleast_1d(np.asarray(anomaly_windows, dtype=int))
            idx = idx[(idx >= 0) & (idx < len(phi))]
            mask[idx] = True
        elif self.threshold is not None:
            mask = phi > self.threshold

        if not mask.any():
            mask[int(np.argmax(phi))] = True
        return mask

    def _build_reconstruction_df(self, recon_arr, variable_names, df_test, timestamps, clip_physical=True):
        """
        Denormalizes (inverse Z-score) the reconstructed values and aligns each window to
        its original timestamp. Optionally clips to the physical lower bound observed in
        the normal training data (never to NaN/missing variables).
        """
        recon_arr = np.asarray(recon_arr, dtype=float).copy()
        means = self.generator.mean.reindex(variable_names).values
        stds = self.generator.std.reindex(variable_names).values
        recon_arr = (recon_arr * stds) + means

        if clip_physical:
            floors = np.array([self.generator.min_physical_vals.get(col, -np.inf) for col in variable_names])
            recon_arr = np.maximum(recon_arr, floors)

        reconstruction_df = pd.DataFrame(recon_arr, columns=list(variable_names))

        if timestamps is not None:
            w = self.generator.w
            s = self.generator.step
            ts_values = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)

            valid_indices = [t - 1 for t in range(w, len(df_test) + 1, s)]
            min_len = min(len(reconstruction_df), len(valid_indices))
            reconstruction_df = reconstruction_df.iloc[:min_len].copy()
            valid_indices = valid_indices[:min_len]

            aligned_timestamps = []
            for idx in valid_indices:
                if idx < len(ts_values):
                    aligned_timestamps.append(ts_values[idx])
                elif aligned_timestamps:
                    aligned_timestamps.append(aligned_timestamps[-1])

            reconstruction_df = reconstruction_df.iloc[:len(aligned_timestamps)].copy()
            reconstruction_df.index = aligned_timestamps
            reconstruction_df.index.name = 'timestamp'
            reconstruction_df.reset_index(inplace=True)

        return reconstruction_df

    def _variable_contribution_scores(self, df_test, batch_size=128, combine_value=True,
                                      anomaly_windows=None):
        """
        Core of the Root Cause Analysis.

        For every generated window it computes each variable's reconstruction error in three
        channels (matrix diagonal / matrix off-diagonal / raw value), STANDARDIZES each
        against that variable's own normal baseline (median + MAD from training), keeps only
        the positive excess, then aggregates ONLY over the windows that belong to the
        detected anomaly.

        This answers the real question -- "which variables deviated from their own normal
        behavior during the anomaly" -- instead of "which variables have large absolute
        error over the whole period" (which just re-discovers chronically noisy sensors).
        The diagonal (self) channel is weighted above the off-diagonal (relational) one so a
        single anomalous variable does not smear blame onto its correlated partners.

        Returns a dict with: variable_names, contrib (Σ excess over anomalous windows),
        peak_z (max standardized excess over those windows), phi, anom_mask, recon_vals.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first!")
        if self.mat_diag_med_ is None:
            raise ValueError("Per-variable baseline not calibrated. Re-run .fit().")

        self.model.eval()

        # Keep only the columns the model was trained on, in the trained order.
        df_test = df_test[self.generator.mean.index]

        try:
            tensor_matrices, tensor_values = self.generator.generate(df_test)
        except ValueError as e:
            raise ValueError(f"Generation error: {e}")
        if tensor_matrices.nelement() == 0:
            raise ValueError("No matrices generated from input dataframe!")

        mat_diag, mat_off, val_err, recon_vals = self._get_variable_errors(
            tensor_matrices, tensor_values, batch_size=batch_size, return_recon=True
        )

        # Per-window phi, IDENTICAL to the detector's COMBINED score (matrix + value, weighted
        # by the learned precisions), so the windows selected here match those that tripped
        # the threshold. Full matrix error = diag.sum + off.sum/2 (off counted from both ends).
        p_mat, p_val = self._task_precisions()
        mat_sumsq = mat_diag.sum(axis=1) + mat_off.sum(axis=1) / 2.0
        val_sumsq = val_err.sum(axis=1)
        phi = p_mat * mat_sumsq + p_val * val_sumsq

        # Standardized positive excess over each variable's own normal baseline, per channel.
        s_diag = np.clip((mat_diag - self.mat_diag_med_) / self.mat_diag_mad_, 0, None)
        s_off = np.clip((mat_off - self.mat_off_med_) / self.mat_off_mad_, 0, None)
        # Lean on the clean self-signal (diagonal); treat the smeared relational error as
        # weaker corroborating evidence so correlated-but-innocent variables are not blamed.
        s_mat = self.rca_self_weight * s_diag + self.rca_rel_weight * s_off
        if combine_value:
            s_val = np.clip((val_err - self.val_err_med_) / self.val_err_mad_, 0, None)
            s = s_mat + s_val
        else:
            s = s_mat

        anom_mask = self._select_anomaly_windows(phi, anomaly_windows)

        contrib = s[anom_mask].sum(axis=0)     # total standardized excess during the anomaly
        peak_z = s[anom_mask].max(axis=0)      # strongest single-window deviation

        del tensor_matrices, tensor_values, mat_diag, mat_off, val_err
        gc.collect()

        return {
            'variable_names': self.generator.mean.index,
            'contrib': contrib,
            'peak_z': peak_z,
            'phi': phi,
            'anom_mask': anom_mask,
            'recon_vals': recon_vals,
        }

    def contribution(self, df_test, df_sistema, timestamps=None, batch_size=128,
                     anomaly_windows=None, z_thresh=3.0, combine_value=True, top_k=None):
        """
        Root Cause Analysis (RCA) pipeline (unified).

        Scoring is identical in both modes: each variable's reconstruction error is
        standardized against its OWN normal baseline (median + MAD from training) and
        aggregated only over the windows that belong to the detected anomaly. The two
        modes differ only in HOW the final variable list is selected, via `top_k`:

        - top_k is None  -> statistical selection. A variable is kept ONLY if it satisfies
            BOTH conditions:
              (A) self-deviation: its peak standardized excess exceeds `z_thresh`
                  (deviates from its OWN normal during the anomaly); and
              (B) peer-standout: its aggregated (already baseline-normalized) contribution
                  is a robust outlier among all variables (MAD rule across variables).
            Requiring both excludes chronically noisy sensors that cross their own baseline
            by chance (fail B) and uniform drifts where nothing truly stands out (fail A).
            If the anomaly is too subtle for the combined cut, it falls back to the top-3.

        - top_k is an int -> fixed cut. Returns the `top_k` variables ranked by contribution
            during the anomaly (no statistical filtering), and the reconstruction DataFrame
            is filtered down to exactly those variables (dashboard usage).

        Returns (contributions_dict, reconstruction_df) in both modes.
        """
        res = self._variable_contribution_scores(
            df_test, batch_size=batch_size, combine_value=combine_value,
            anomaly_windows=anomaly_windows
        )
        variable_names = res['variable_names']
        contrib = res['contrib']
        peak_z = res['peak_z']
        n_features = len(variable_names)

        df_contrib = pd.DataFrame({
            'VARIAVEL': variable_names,
            'score': contrib,
            'peak_z': peak_z,
        })
        df_contrib = pd.merge(df_contrib, df_sistema[['VARIAVEL', 'DESC', 'SISTEMA']], on='VARIAVEL', how='left')
        df_contrib['DESC'] = df_contrib['DESC'].fillna('NoDesc')
        df_contrib['SISTEMA'] = df_contrib['SISTEMA'].fillna('NoSystem')

        if top_k is None:
            # --- Statistical mode: (A) self-deviation AND (B) peer-standout ---
            cond_self = peak_z > z_thresh
            k = 1.4826
            c_med = np.median(contrib)
            c_mad = np.median(np.abs(contrib - c_med)) * k + 1e-9
            cond_peer = contrib > (c_med + z_thresh * c_mad)
            is_contrib = cond_self & cond_peer
            if not is_contrib.any():
                # Anomaly too subtle for the combined cut: fall back to the top-3.
                top = np.argsort(contrib)[::-1][:3]
                is_contrib = np.zeros(n_features, dtype=bool)
                is_contrib[top] = True
            df_contrib = df_contrib[is_contrib].copy()
            df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)
        else:
            # --- Top-K mode: fixed cut by ranked contribution ---
            df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)
            df_contrib = df_contrib.head(int(top_k)).copy()

        total = df_contrib['score'].sum()
        df_contrib['%'] = (df_contrib['score'] / total * 100) if total > 0 else 0.0

        df_contrib.index = df_contrib.index.astype(str)
        df_contrib = df_contrib[['VARIAVEL', 'DESC', 'SISTEMA', 'score', 'peak_z', '%']]
        contributions_dict = df_contrib.to_dict()

        reconstruction_df = self._build_reconstruction_df(
            res['recon_vals'], variable_names, df_test, timestamps, clip_physical=True
        )

        # In top-K mode, send only the selected reconstructed variables to the panel.
        if top_k is not None:
            selected_vars = df_contrib['VARIAVEL'].tolist()
            keep_cols = (['timestamp'] if 'timestamp' in reconstruction_df.columns else []) + selected_vars
            reconstruction_df = reconstruction_df[keep_cols].copy()

        return contributions_dict, reconstruction_df