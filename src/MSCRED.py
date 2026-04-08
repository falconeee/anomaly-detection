import pandas as pd
import numpy as np
import time
import math
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Helper Classes (Models & Utils)
def set_deterministic(seed=42):
    """
    Enforces reproducibility across multiple runs by fixing seeds for Python, 
    NumPy, PyTorch, and CUDA backend operations.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Disables auto-tuner to prioritize determinism over performance speedups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM. Replaces standard fully-connected layers with 2D 
    convolutions to preserve spatial structures (e.g., sensor correlations) 
    while learning temporal dynamics.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # Padding ensures the output spatial dimensions (H, W) match the input
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # A single convolution layer computes all 4 gates simultaneously (4 * hidden_dim)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and previous hidden state along the channel dimension (dim=1)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into the 4 LSTM gates (Input, Forget, Output, Cell Candidate)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply standard LSTM non-linearities
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        """Initializes hidden and cell states with zeros for the first time step."""
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class TemporalAttention(nn.Module):
    """
    Dot-product Temporal Attention mechanism. 
    Uses the last time step as a query to attend to all past historical states, 
    building a context vector that mitigates the LSTM's forgetting issue over long sequences.
    """
    def __init__(self):
        super(TemporalAttention, self).__init__()

    def forward(self, hidden_states):
        # Input shape: [batch, steps, channels, H, W]
        last_step = hidden_states[:, -1, ...]
        steps = hidden_states.shape[1]
        weights = []
        
        # Calculate dot-product similarity between each historical step and the last step
        for k in range(steps):
            curr_step = hidden_states[:, k, ...]
            # Sum over Channels, Height, and Width to get a single scalar score per batch element
            dot_product = torch.sum(curr_step * last_step, dim=(1, 2, 3)) 
            
            # Division by 'steps' acts as a scaling factor to prevent exploding logits before softmax
            weights.append(dot_product / steps)
        
        # Stack temporal weights to shape [batch, steps]
        weights = torch.stack(weights, dim=1)
        
        # Apply softmax to normalize scores into probability weights
        # Unsqueeze dimensions to broadcast the scalar weight across C, H, W during multiplication
        weights = F.softmax(weights, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Compute the final context vector as the weighted sum of all historical states
        context = torch.sum(hidden_states * weights, dim=1)
        return context

class MSCRED(nn.Module):
    """
    Hybrid Multivariate Spatial-Temporal Convolutional Recurrent Network.
    Simultaneously reconstructs sensor correlation matrices and raw magnitude values.
    """
    def __init__(self, sensor_n, scale_n, step_max):
        super(MSCRED, self).__init__()
        self.sensor_n = sensor_n
        
        # Spatial Encoder: Extracts features from multi-scale correlation matrices
        self.conv1 = nn.Conv2d(scale_n, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        
        # Temporal Encoder: Preserves 2D spatial structures while learning time dependencies
        self.lstm1 = ConvLSTMCell(32, 32, (3,3), True)
        self.lstm2 = ConvLSTMCell(64, 64, (3,3), True)
        self.lstm3 = ConvLSTMCell(128, 128, (3,3), True)
        self.lstm4 = ConvLSTMCell(256, 256, (3,3), True)
        
        # Context Aggregator: Mitigates LSTM forgetting over long sequences
        self.attention = TemporalAttention()
        
        # Spatial Decoder: Upsamples latent representation back to matrix dimensions
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, scale_n, kernel_size=3, stride=1, padding=1)
        
        # Hybrid Value Head: MLP for raw value regression (magnitude modeling)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), # Stabilizes continuous regression
            nn.GELU(),           # Prevents dying gradients for negative scaled inputs
            nn.Linear(256, sensor_n)
        )
        
    def forward(self, x):
        b, steps, c, h, w = x.size()
    
        # Dynamic spatial shape inference for ConvLSTM initialization
        dummy_in = torch.zeros(1, c, h, w, device=x.device)
        d1 = self.conv1(dummy_in); d2 = self.conv2(d1); d3 = self.conv3(d2); d4 = self.conv4(d3)
        
        state1 = self.lstm1.init_hidden(b, (d1.size(2), d1.size(3)), x.device)
        state2 = self.lstm2.init_hidden(b, (d2.size(2), d2.size(3)), x.device)
        state3 = self.lstm3.init_hidden(b, (d3.size(2), d3.size(3)), x.device)
        state4 = self.lstm4.init_hidden(b, (d4.size(2), d4.size(3)), x.device)

        h1_list, h2_list, h3_list, h4_list = [], [], [], []

        # Step-by-step spatial-temporal encoding
        for t in range(steps):
            input_t = x[:, t, :, :, :]
            # SELU activation provides self-normalization, stabilizing deep architectures
            enc1 = F.selu(self.conv1(input_t))
            enc2 = F.selu(self.conv2(enc1))
            enc3 = F.selu(self.conv3(enc2))
            enc4 = F.selu(self.conv4(enc3))
            
            h1, c1 = self.lstm1(enc1, state1); state1 = (h1, c1); h1_list.append(h1)
            h2, c2 = self.lstm2(enc2, state2); state2 = (h2, c2); h2_list.append(h2)
            h3, c3 = self.lstm3(enc3, state3); state3 = (h3, c3); h3_list.append(h3)
            h4, c4 = self.lstm4(enc4, state4); state4 = (h4, c4); h4_list.append(h4)
            
        # Apply temporal attention to aggregate historical states into single context maps
        h1_stack = torch.stack(h1_list, dim=1); attn1 = self.attention(h1_stack)
        h2_stack = torch.stack(h2_list, dim=1); attn2 = self.attention(h2_stack)
        h3_stack = torch.stack(h3_list, dim=1); attn3 = self.attention(h3_stack)
        h4_stack = torch.stack(h4_list, dim=1); attn4 = self.attention(h4_stack)
        
        # Decoder with Skip Connections: Concatenates attention maps to preserve local details
        # F.interpolate safely handles spatial dimension mismatches caused by odd-sized inputs
        dec4 = F.selu(self.deconv4(attn4))
        if dec4.shape[2:] != attn3.shape[2:]: dec4 = F.interpolate(dec4, size=attn3.shape[2:])
        dec4_concat = torch.cat([dec4, attn3], dim=1)
        
        dec3 = F.selu(self.deconv3(dec4_concat))
        if dec3.shape[2:] != attn2.shape[2:]: dec3 = F.interpolate(dec3, size=attn2.shape[2:])
        dec3_concat = torch.cat([dec3, attn2], dim=1)
        
        dec2 = F.selu(self.deconv2(dec3_concat))
        if dec2.shape[2:] != attn1.shape[2:]: dec2 = F.interpolate(dec2, size=attn1.shape[2:])
        dec2_concat = torch.cat([dec2, attn1], dim=1)
        
        # Route 1: Standard matrix reconstruction (Correlation Topology)
        matrix_output = F.selu(self.deconv1(dec2_concat))
        
        # Route 2: Hybrid value regression (Magnitude Spikes)
        # Uses the deepest contextual representation (attn4)
        latent_features = self.global_pool(attn4)
        value_output = self.value_head(latent_features)
        
        return matrix_output, value_output

class HybridDataset(Dataset):
    """
    Feeds standard input matrices (X) alongside target regression values (Y).
    """
    def __init__(self, matrix_array, value_array):
        self.matrices = matrix_array
        self.values = value_array

    def __len__(self):
        return self.matrices.shape[0]

    def __getitem__(self, idx):
        # Conversion to float32 ensures precision compatibility with PyTorch weights
        return (torch.from_numpy(self.matrices[idx]).float(), 
                torch.from_numpy(self.values[idx]).float())

class SPOT:
    """
    Streaming Peaks-Over-Threshold (SPOT) Algorithm.
    Applies Extreme Value Theory (EVT) to dynamically set robust anomaly thresholds.
    Instead of assuming a Gaussian distribution, it models the 'tail' (extreme values) 
    using a Generalized Pareto Distribution (GPD).
    """
    def __init__(self, q=1e-4):
        # q: The desired probability of false alarms (Risk parameter).
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, init_data, data):
        """Loads and splits data into calibration (init_data) and streaming (data) sets."""
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
        """Appends new streaming data."""
        if isinstance(data, list): data = np.array(data)
        elif isinstance(data, np.ndarray): data = data
        elif isinstance(data, pd.Series): data = data.values
        else: return
        self.data = np.append(self.data, data)

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Calibration phase: Establishes a base threshold (e.g., 98th percentile).
        Values above this threshold are extracted as 'peaks' to fit the GPD.
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level
            
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        
        # Base threshold (t) and peaks (Y)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
        
        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')
            
        # Fit GPD parameters
        g, s, l = self._grimshaw()
        # Calculate final mathematical alarm threshold
        self.extreme_quantile = self._quantile(g, s)
        
        if verbose:
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
        """Finds roots for the derivative equation using SciPy optimization."""
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

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        """Calculates the log-likelihood of the Generalized Pareto Distribution."""
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + math.log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Grimshaw's Trick (1993): Highly efficient numerical method to find the 
        Maximum Likelihood Estimation (MLE) for GPD parameters (gamma and sigma).
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
        Inverse Cumulative Distribution Function (CDF) for the GPD.
        Calculates the final threshold value for the desired risk level (proba).
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0: 
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: 
            return self.init_threshold - sigma * math.log(r)

class MSCRED:
    """
    Main orchestration class for the Hybrid MSCRED pipeline.
    Manages configuration, data preprocessing, and training/inference workflows.
    """
    def __init__(self):
        # Centralized hyperparameters dictionary makes it easy to save/load configurations
        self.model_config = {
            'batch_size': 128,
            'win_size': [10, 30, 60], # Multi-scale temporal contexts
            'step_max': 5,            # LSTM sequence length
            'gap_time': 1,            # Sliding window stride
            'learning_rate': 0.0003,
            'epochs': 5,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        self.device = self.model_config['device']
        self.model = None
        self.scaler_params = None
        self.threshold = None
        self.gain = 1.0 # Default sensitivity multiplier for the EVT/POT threshold
        
        # Enforces absolute reproducibility
        set_deterministic()

    def _get_scaler(self, df_list):
        """
        Calculates global MinMax boundaries for each individual sensor across all training blocks.
        Vital for keeping matrix inner-products bounded and stable.
        """
        full_df = pd.concat(df_list, axis=0)
        # Transpose (T) aligns dimensions to (Sensors, Timestamps)
        data = np.array(full_df.values.T, dtype=np.float64)
        
        # keepdims=True ensures shapes remain (Sensors, 1) for easy broadcasting later
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        return min_val, max_val

    def _generate_signature_matrix(self, df_block):
        """
        Core data transformation pipeline.
        Converts flat multivariate time-series into 5D spatial-temporal tensors:
        (Samples, Time_Steps, Scales, Sensors, Sensors)
        """
        step_max = self.model_config['step_max']
        gap_time = self.model_config['gap_time']
        win_size = self.model_config['win_size']
        
        data = np.array(df_block.values.T, dtype=np.float64)
        sensor_n = data.shape[0]
        total_time = data.shape[1]
        
        # 1. Normalization
        min_val, max_val = self.scaler_params
        epsilon = 1e-6
        data = (data - min_val) / (max_val - min_val + epsilon)

        # 2. Multi-scale Matrix Generation (Inner Products)
        # Calculates the spatial correlation between all sensor pairs within each window
        data_all_scales = []
        for w in range(len(win_size)):
            matrix_list = []
            win = win_size[w]
            
            for t in range(0, total_time, gap_time):
                matrix_t = np.zeros((sensor_n, sensor_n))
                if t >= win:
                    segment = data[:, t - win : t]
                    # Matrix Multiplication: (N x W) * (W x N) = (N x N)
                    matrix_t = np.matmul(segment, segment.T) / win
                matrix_list.append(matrix_t)
            data_all_scales.append(matrix_list)

        # 3. Temporal Sequence Assembly for ConvLSTM
        X_block = []
        total_generated_steps = len(data_all_scales[0])
        num_scales = len(win_size)
        
        # Ensures we only start creating sequences when enough historical matrices exist
        start_idx = (win_size[-1] // gap_time) + step_max

        for i in range(total_generated_steps):
            if i < start_idx:
                continue
            
            sequence_matrices = [] 
            # Collects the last 'step_max' matrices to form the temporal sequence
            for step in range(step_max, 0, -1):
                idx = i - step
                multi_scale_step = []
                # Stacks the different scales (Channels) for this specific time step
                for scale_idx in range(num_scales):
                    multi_scale_step.append(data_all_scales[scale_idx][idx])
                sequence_matrices.append(multi_scale_step)
            
            X_block.append(np.array(sequence_matrices))

        if len(X_block) > 0:
            return np.array(X_block)
        else:
            return np.empty((0))

    def _prepare_hybrid_data(self, X_matrices, df_source):
        """
        Aligns raw target values with the generated correlation matrices.
        Essential for the Hybrid architecture (Regression Head).
        """
        gap_time = self.model_config['gap_time']
        win_size = self.model_config['win_size']
        step_max = self.model_config['step_max']
        
        # Calculates the exact index drop offset caused by the sliding window buffer
        start_gap_steps = (win_size[-1] // gap_time) + step_max
        start_index_real = start_gap_steps * gap_time
        
        min_val, max_val = self.scaler_params
        data_values = df_source.values
        
        # Applies global MinMax scaling identically to the matrices
        data_norm = (data_values - min_val.T) / (max_val.T - min_val.T + 1e-6)
        
        # Temporal slicing to perfectly match row 0 of Y with row 0 of X
        values_aligned = data_norm[start_index_real :: gap_time]
        
        # Safety cutoff to prevent length mismatch errors 
        min_len = min(len(X_matrices), len(values_aligned))
        
        X_final = X_matrices[:min_len]
        y_final = values_aligned[:min_len]
        
        return X_final, y_final

    def fit(self, df_train_list, gain=1, epochs=None):
        # Support for both single DataFrame or lists (useful for disjoint periods)
        if not isinstance(df_train_list, list):
            df_train_list = [df_train_list]
            
        self.gain = gain
        if epochs is not None:
             self.model_config['epochs'] = epochs

        # Scaler
        self.scaler_params = self._get_scaler(df_train_list)
        
        # Generate Matrices
        X_train_list = []
        for i, block in enumerate(df_train_list):
            X_proc = self._generate_signature_matrix(block)
            if X_proc.size > 0:
                X_train_list.append(X_proc)
        
        if not X_train_list:
            raise ValueError("No training data generated.")
            
        X_train_final = np.concatenate(X_train_list, axis=0)
        
        # Prepare Hybrid Data (Aligns X matrices with Y raw values)
        df_train_full = pd.concat(df_train_list)
        X_train_hybrid, y_train_hybrid = self._prepare_hybrid_data(X_train_final, df_train_full)
        
        # Initialize Model dynamically based on generated data shapes
        sample_shape = X_train_hybrid.shape
        scale_n = sample_shape[2]
        self.sensor_n = sample_shape[3]
        
        self.model = MSCRED(sensor_n=self.sensor_n, scale_n=scale_n, step_max=self.model_config['step_max']).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training DataLoader
        dataset = HybridDataset(X_train_hybrid, y_train_hybrid)
        # shuffle=True is critical here to break temporal sequence bias during batch training
        dataloader = DataLoader(dataset, batch_size=self.model_config['batch_size'], shuffle=True) 
        
        self.model.train()
        print(f"Starting training for {self.model_config['epochs']} epochs...")
        
        for epoch in range(self.model_config['epochs']):
            start_time = time.time()
            epoch_loss = 0
            for batch_matrix, batch_values in dataloader:
                batch_matrix = batch_matrix.to(self.device)
                batch_values = batch_values.to(self.device)
                
                # Ground truth for matrix is the last temporal step in the sequence
                target_matrix = batch_matrix[:, -1, :, :, :]
                
                optimizer.zero_grad()
                recon_matrix, recon_values = self.model(batch_matrix)
                
                loss_matrix = criterion(recon_matrix, target_matrix)
                loss_value = criterion(recon_values, batch_values)

                # Dimensionality Balancing: Scales the mean errors by the number of elements
                # Ensures the Value Loss (N elements) isn't overpowered by the Matrix Loss (N*N elements)
                n_elements = self.sensor_n * self.sensor_n
                loss_matrix_scaled = loss_matrix * n_elements
                loss_value_scaled = loss_value * n_elements

                # Alpha controls the weight of the hybrid value regression
                alpha = 1.0 
                total_loss = loss_matrix_scaled + (alpha * loss_value_scaled)

                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            print(f"Epoch {epoch+1}/{self.model_config['epochs']} | Loss: {epoch_loss/len(dataloader):.6f} | Time: {time.time() - start_time:.2f}s")

        # Calculate Threshold (POT Calibration)
        self.model.eval()
        train_scores = self._get_train_scores(X_train_hybrid, y_train_hybrid)
        self.threshold = self._pot_eval(train_scores) * self.gain
        print(f"Threshold: {self.threshold}")
        
    def _get_train_scores(self, X_train, y_train):
        # NOTE: Using np.float32 instead of float64 here saves 50% RAM and is sufficient for PyTorch
        dataset = HybridDataset(X_train.astype(np.float32), y_train.astype(np.float32))
        loader = DataLoader(dataset, batch_size=32, shuffle=False) # shuffle=False to maintain temporal order
        
        all_scores = []
        with torch.no_grad():
            for batch_x, _ in loader:
                # We only need the matrix error to establish the anomaly threshold
                recon_matrix, _ = self.model(batch_x.to(self.device))
                
                # Grabs the last step directly from the CPU batch to avoid unnecessary GPU-to-CPU transfer
                gt_batch = batch_x.numpy()[:, -1, :, :, :]
                recon_np = recon_matrix.cpu().numpy()
                
                # Calculate MSE across spatial dimensions (Scales, H, W)
                diff = np.square(gt_batch - recon_np)
                scores = np.mean(diff, axis=(1, 2, 3))
                all_scores.append(scores)
                
        return np.concatenate(all_scores, axis=0)

    def _pot_eval(self, init_score, q=1e-4, level=0.02):
        """
        Calculates the Extreme Value Theory threshold.
        Includes a fallback mechanism that smoothly lowers the peak threshold 
        if the SciPy optimization fails to converge.
        """
        lms = level
        while True:
            try:
                s = SPOT(q)
                s.fit(init_score, init_score)
                s.initialize(level=lms, min_extrema=False, verbose=False)
                return s.extreme_quantile
            except Exception:
                # Graceful degradation of the peak parameter
                lms = lms * 0.999

    def predict(self, df_test, timestamps=None):
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if timestamps is None:
            timestamps = df_test.index
            
        X_test = self._generate_signature_matrix(df_test)
        
        # Calculate the temporal offset caused by the sliding window buffer
        gap_time = self.model_config['gap_time']
        win_size = self.model_config['win_size']
        step_max = self.model_config['step_max']
        start_gap_steps = (win_size[-1] // gap_time) + step_max
        start_index_real = start_gap_steps * gap_time
        
        # Align timestamps to match the valid predictions generated
        if hasattr(timestamps, 'values'):
            ts_values = timestamps.values
        else:
            ts_values = np.array(timestamps)
        
        valid_timestamps = ts_values[start_index_real :: gap_time]
        
        # Retrieve both matrices and raw values for hybrid inference
        X_test_final, _ = self._prepare_hybrid_data(X_test, df_test)
        
        # Dummy targets used since we only need the input for prediction
        dataset = HybridDataset(X_test_final.astype(np.float64), np.zeros((len(X_test_final), self.sensor_n)))
        loader = DataLoader(dataset, batch_size=self.model_config['batch_size'], shuffle=False)
        
        self.model.eval()

        matrix_scores = []
        value_scores = []

        with torch.no_grad():
            # Loader unpacks both X (matrices) and Y (dummy targets, ignored here)
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y_np = batch_y.numpy()
                
                recon_matrix, recon_vals = self.model(batch_x)
                
                # 1. Matrix Score (Topology/Correlation Error)
                # Compares the reconstructed matrix with the last step of the input sequence
                gt_matrix = batch_x.cpu().numpy()[:, -1, :, :, :]
                recon_matrix_np = recon_matrix.cpu().numpy()
                diff_matrix = np.square(gt_matrix - recon_matrix_np)
                
                # Averages over spatial dimensions (Scales, Height, Width) -> Shape: (Batch,)
                scores_m = np.mean(diff_matrix, axis=(1, 2, 3)) 
                
                # 2. Value Score (Hybrid Magnitude Error)
                recon_vals_np = recon_vals.cpu().numpy()
                diff_vals = np.square(batch_y_np - recon_vals_np)
                
                # Averages over sensors -> Shape: (Batch,)
                scores_v = np.mean(diff_vals, axis=1) 
                
                matrix_scores.append(scores_m)
                value_scores.append(scores_v)

        final_matrix_scores = np.concatenate(matrix_scores, axis=0)
        final_value_scores = np.concatenate(value_scores, axis=0)

        # Dimensional Balance: Since both scores are means, they are on the same scale.
        # Alpha controls the weight of the raw value anomaly relative to the correlation anomaly.
        alpha = 1.0
        final_scores = final_matrix_scores + (alpha * final_value_scores)
        
        # Safe length alignment to prevent index out of bounds
        min_len = min(len(final_scores), len(valid_timestamps))
        
        return {
            'timestamp': valid_timestamps[:min_len],
            'phi': final_scores[:min_len]
        }

    def contribution(self, df_test, df_sistema, timestamps=None, batch_size=32, alpha=1.0):
        """
        Root Cause Analysis (RCA) for Hybrid MSCRED.
        Combines spatial correlation errors (matrices) and magnitude errors (values).
        Uses MAD (Median Absolute Deviation) to dynamically filter root cause variables.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
            
        self.model.eval()
        
        X_test = self._generate_signature_matrix(df_test)
        
        if len(X_test) == 0:
            raise ValueError("No matrices generated from input dataframe!")
            
        X_test_final, y_test_final = self._prepare_hybrid_data(X_test, df_test)
        
        dataset = HybridDataset(X_test_final.astype(np.float32), y_test_final.astype(np.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Accumulators for total absolute errors
        total_error_matrix = np.zeros((self.sensor_n, self.sensor_n), dtype=np.float64)
        total_error_val = np.zeros(self.sensor_n, dtype=np.float64)
        
        reconstructed_values_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y_np = batch_y.numpy() 
                
                recon_matrix, recon_vals = self.model(batch_x)
                
                # --- Matrix Error (Broken Correlations) ---
                recon_matrix_np = recon_matrix.cpu().numpy()
                gt_batch_mat = batch_x.cpu().numpy()[:, -1, :, :, :]
                
                diff_mat = np.square(gt_batch_mat - recon_matrix_np)
                diff_aggregated_scales = np.mean(diff_mat, axis=1) 
                
                # Sums error over the batch to accumulate the total period contribution
                total_error_matrix += np.sum(diff_aggregated_scales, axis=0) 
                
                # --- Value Error (Direct Hybrid Magnitude) ---
                recon_vals_np = recon_vals.cpu().numpy()
                diff_val = np.square(batch_y_np - recon_vals_np)
                total_error_val += np.sum(diff_val, axis=0) 
                
                # Stores values for later physical denormalization
                reconstructed_values_list.extend(recon_vals_np)

        # Score Processing & Dimensional Balance
        # Summing dim=1 calculates how much sensor 'i' broke correlations with ALL other sensors
        mat_scores = np.sum(total_error_matrix, axis=1)
        val_scores = total_error_val
        
        # The matrix row has N elements, the value vector has 1.
        # Multiplying by N scales the value error so it has equal voting weight.
        val_scores_scaled = val_scores * self.sensor_n
        variable_scores = mat_scores + (alpha * val_scores_scaled)
        
        variable_names = df_test.columns
        total_period_error = np.sum(variable_scores)
        
        contrib_pct = (variable_scores / total_period_error * 100) if total_period_error > 0 else np.zeros_like(variable_scores)

        # Construct raw Contribution DataFrame
        df_contrib = pd.DataFrame({
            'VARIAVEL': variable_names,
            'score': variable_scores,
            '%': contrib_pct
        })
        
        # Merge physical descriptions and system data safely
        cols_to_merge = ['VARIAVEL', 'DESC']
        if 'SISTEMA' in df_sistema.columns:
            cols_to_merge.append('SISTEMA')
            
        df_contrib = pd.merge(df_contrib, df_sistema[cols_to_merge], on='VARIAVEL', how='left')
        df_contrib['DESC'] = df_contrib['DESC'].fillna('NoDesc')
        if 'SISTEMA' in df_contrib.columns:
            df_contrib['SISTEMA'] = df_contrib['SISTEMA'].fillna('NoSystem')
        
        df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)

        # --- Dynamic Outlier Identification (MAD Approach) ---
        # Robust against "Masking Effect" caused by extreme, systemic anomalies.
        df_contrib_backup = df_contrib.copy()
        
        median_score = df_contrib['score'].median()
        mad = (df_contrib['score'] - median_score).abs().median()
        
        # k=1.4826 scales the MAD to approximate standard deviation behavior
        k = 1.4826
        mad_threshold = median_score + (k * mad)
        
        # Isolate sensors acting as mathematical outliers (Root Causes)
        df_contrib = df_contrib[df_contrib['score'] > mad_threshold].copy()
        
        # Fallback: Enforce top 3 culprits if anomaly signature is too subtle
        if len(df_contrib) == 0:
            df_contrib = df_contrib_backup.head(3).copy()

        # Recalculate relative culpability strictly within the isolated culprit group
        if df_contrib['score'].sum() > 0:
            df_contrib['%'] = (df_contrib['score'] / df_contrib['score'].sum()) * 100
        else:
            df_contrib['%'] = 0.0

        # Output Formatting
        df_contrib.index = df_contrib.index.astype(str)
        
        final_cols = ['VARIAVEL', 'DESC', 'score', '%']
        if 'SISTEMA' in df_contrib.columns: final_cols.insert(2, 'SISTEMA')
        
        df_contrib = df_contrib[final_cols]
        contributions_dict = df_contrib.to_dict(orient='dict')

        # --- Denormalization for Physical Dashboards ---
        min_val, max_val = self.scaler_params
        recon_arr = np.array(reconstructed_values_list)
        
        # Inverse MinMax Transform: Val = Norm * (Max - Min) + Min
        recon_real = recon_arr * (max_val.T - min_val.T) + min_val.T
        
        # Temporal Alignment for Reconstruction View
        if timestamps is not None:
            gap_time = self.model_config['gap_time']
            win_size = self.model_config['win_size']
            step_max = self.model_config['step_max']
            
            start_gap_steps = (win_size[-1] // gap_time) + step_max
            start_index_real = start_gap_steps * gap_time
            
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)
                
            valid_timestamps = ts_values[start_index_real :: gap_time]
            
            min_len = min(len(recon_real), len(valid_timestamps))
            recon_real = recon_real[:min_len]
            valid_timestamps = valid_timestamps[:min_len]
            
            df_reconstruction = pd.DataFrame(recon_real, columns=df_test.columns, index=valid_timestamps)
            df_reconstruction.index.name = 'timestamp'
            df_reconstruction.reset_index(inplace=True)
        else:
            df_reconstruction = pd.DataFrame(recon_real, columns=df_test.columns)
            
        return contributions_dict, df_reconstruction