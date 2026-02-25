import pandas as pd
import numpy as np
import math
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Helper Classes
class AttributeMatrixGenerator:
    def __init__(self, window_size=10, step=10):
        self.w = window_size
        self.step = step
        self.mean = None
        self.std = None

    def fit_scaler(self, train_dataframes):
        if isinstance(train_dataframes, pd.DataFrame):
            train_dataframes = [train_dataframes]
        # Concat to calculate global statistics
        full_train_df = pd.concat(train_dataframes, ignore_index=True)
        self.mean = full_train_df.mean()
        self.std = full_train_df.std() + 1e-6

    def generate(self, df):
        if self.mean is None:
            raise ValueError("Execute .fit_scaler() primeiro.")

        # Scaling
        data = (df - self.mean) / self.std
        values = np.nan_to_num(data.values)
        
        matrices = []
        target_values = [] 

        if len(values) < self.w:
            # Return empty tuple if df is too small (i.e. smaller than window_size)
            return torch.empty(0), torch.empty(0)

        for t in range(self.w, len(values), self.step):
            x_segment = values[t-self.w : t] 
            
            # Matrix (Eq. 1)
            x_t = torch.tensor(x_segment, dtype=torch.float32).T
            m_t = torch.matmul(x_t, x_t.T) / self.w
            matrices.append(m_t)
            
            last_val = torch.tensor(x_segment[-1], dtype=torch.float32)
            target_values.append(last_val)
            
        if not matrices:
            return torch.empty(0), torch.empty(0)

        # Tuple: (Tensor of Matrices, Tensor of Values)
        return torch.stack(matrices).unsqueeze(1), torch.stack(target_values)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.scale = 5.0 # Attention mechanism (how many windows the algorithm looks at to perform the current prediction)

    def forward(self, h_current, h_history):
        if not h_history:
            return h_current
        
        context = torch.stack(h_history, dim=0)
        b, c, h, w = h_current.size()
        flat_curr = h_current.view(b, -1).unsqueeze(1) 
        flat_hist = context.view(context.size(0), b, -1).permute(1, 2, 0) 
        
        scores = torch.bmm(flat_curr, flat_hist) / self.scale 
        weights = F.softmax(scores, dim=-1) 
        
        weighted_hist = torch.bmm(weights, flat_hist.permute(0, 2, 1))
        h_hat = weighted_hist.view(b, c, h, w)
        
        return h_hat

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class MSCVAE_Hybrid(nn.Module):
    def __init__(self, n_features):
        super(MSCVAE_Hybrid, self).__init__()
        self.n_features = n_features
        latent_dim=round(math.sqrt(n_features))
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.enc2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.enc3 = nn.Conv2d(32, 64, 3, 2, 1)
        
        # Dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features, n_features)
            out = self.enc3(self.enc2(self.enc1(dummy)))
            self.flatten_dim = out.view(1, -1).size(1)
            self.spatial_shape = out.shape[1:] 

        # Latent
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder Matrix
        self.fc_decode_mat = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder of Values (Hybrid)
        self.val_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_features) 
        )

        # Temporal Modeling
        self.clstm = ConvLSTMCell(64, 64, kernel_size=3, bias=True)
        self.attention = TemporalAttention(hidden_dim=64)
        
        # Convolutional Decoder
        self.dec3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1)
        self.dec1 = nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1)
        
        self.h_state = None; self.c_state = None; self.history = []; self.max_history = 5

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        else:
            return mu

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder
        e3 = F.relu(self.enc3(F.relu(self.enc2(F.relu(self.enc1(x))))))
        flat = e3.view(batch_size, -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        
        # Route 1: Values
        recon_values = self.val_decoder(z)
        
        # Route 2: Matrices
        if self.h_state is None or self.h_state.size(0) != batch_size:
            h, w = self.spatial_shape[1], self.spatial_shape[2]
            self.h_state = torch.zeros(batch_size, 64, h, w).to(x.device)
            self.c_state = torch.zeros(batch_size, 64, h, w).to(x.device)
            self.history = []

        h_next, c_next = self.clstm(e3, (self.h_state, self.c_state))
        h_att = self.attention(h_next, self.history)
        
        self.h_state = h_att.detach()
        self.c_state = c_next.detach()
        self.history.append(self.h_state)
        if len(self.history) > self.max_history: self.history.pop(0)

        z_dec = self.fc_decode_mat(z).view(batch_size, *self.spatial_shape)
        combined = z_dec + h_att
        
        d3 = F.relu(self.dec3(combined))
        d2 = F.relu(self.dec2(d3))
        recon_matrix = self.dec1(d2)
        
        if recon_matrix.shape != x.shape:
            recon_matrix = F.interpolate(recon_matrix, size=x.shape[2:], mode='bilinear', align_corners=False)

        return recon_matrix, recon_values, mu, logvar

    def loss_function(self, recon_matrix, x_matrix, recon_values, x_values, mu, logvar, alpha=1):
        MSE_Mat = F.mse_loss(recon_matrix, x_matrix, reduction='sum')
        MSE_Val = F.mse_loss(recon_values, x_values, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return MSE_Mat + (alpha * MSE_Val) + KLD

class SPOT:
    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, init_data, data):
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
        if isinstance(data, list): data = np.array(data)
        elif isinstance(data, np.ndarray): data = data
        elif isinstance(data, pd.Series): data = data.values
        else: return
        self.data = np.append(self.data, data)

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')
        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        if verbose:
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

    def _rootsFinder(fun, jac, bounds, npoints, method):
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
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + math.log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
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
        r = self.n * self.proba / self.Nt
        if gamma != 0: return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: return self.init_threshold - sigma * math.log(r)

    def run(self, with_alarm=True, dynamic=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you should initialize before running again')
            return {}
        th = []
        alarm = []
        for i in range(self.data.size):
            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                if self.data[i] > self.extreme_quantile:
                    if with_alarm: alarm.append(i)
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)
                elif self.data[i] > self.init_threshold:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1
            th.append(self.extreme_quantile)
        return {'thresholds': th, 'alarms': alarm}


# Main Class
class MSCVAE:
    def __init__(self, n_features=None, window_size=10, stride=1, device=None, seed=42):
        self.seed = seed
        self.set_deterministic(self.seed)
        
        self.n_features = n_features
        self.window_size = window_size
        self.stride = stride
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = None
        self.generator = AttributeMatrixGenerator(window_size=self.window_size, step=self.stride)
        self.threshold = None
        self.gain = 1.0 # default gain
    
    def set_deterministic(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, train_data, epochs=50, batch_size=128, lr=1e-3, gain=1.0, verbose=True):
        self.gain = gain
        if isinstance(train_data, pd.DataFrame):
            train_data = [train_data]
            
        # Fit scaler
        if verbose: print("Fitting scaler...")
        self.generator.fit_scaler(train_data)
        
        # Prepare data
        if verbose: print("Generating training data...")
        train_matrices = []
        train_values = []
        
        # If n_features was not set, set it from the first dataframe
        if self.n_features is None:
            self.n_features = train_data[0].shape[1]
            
        for df in train_data:
            t_mat, t_val = self.generator.generate(df)
            if t_mat.nelement() > 0:
                train_matrices.append(t_mat)
                train_values.append(t_val)
        
        if not train_matrices:
             raise ValueError("No training data generated. Check window_size and data length.")

        final_train_matrix = torch.cat(train_matrices, dim=0)
        final_train_values = torch.cat(train_values, dim=0)
        
        train_loader = DataLoader(
            TensorDataset(final_train_matrix, final_train_values), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Initialize Model
        self.model = MSCVAE_Hybrid(n_features=self.n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training
        self.model.train()
        if verbose: print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_matrix, batch_values in train_loader:
                x_mat = batch_matrix.to(self.device)
                x_val = batch_values.to(self.device)
                
                optimizer.zero_grad()
                recon_mat, recon_val, mu, logvar = self.model(x_mat)
                loss = self.model.loss_function(recon_mat, x_mat, recon_val, x_val, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader.dataset):.4f}")

        # Calculate Threshold (POT)
        if verbose: print("Calculating threshold...")
        train_scores = self._get_anomaly_scores(final_train_matrix)
        self.threshold = self._pot_eval(train_scores)
        
        if verbose:
            print(f"Base Threshold (POT): {self.threshold:.6f}")
            print(f"Gain: {self.gain}")
            print(f"Final Threshold: {self.threshold * self.gain:.6f}")

    def _get_anomaly_scores(self, data_tensor, batch_size=128):
        self.model.eval()
        mse_loss = nn.MSELoss(reduction='none')
        scores = []
        
        # Create loader for internal score calculation
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            self.model.h_state = None
            self.model.c_state = None
            self.model.history = []
            
            for (batch_input,) in loader:
                x = batch_input.to(self.device)
                recon_mat, _, _, _ = self.model(x)
                # Score based on matrix reconstruction
                loss = mse_loss(recon_mat, x).sum(dim=(1, 2, 3)).cpu().numpy()
                scores.extend(loss)
                
        return np.array(scores)

    def _pot_eval(self, init_score, q=1e-4, level=0.02):
        lms = level
        while True:
            try:
                s = SPOT(q)
                s.fit(init_score, init_score)
                s.initialize(level=lms, min_extrema=False, verbose=False)
            except Exception:
                lms = lms * 0.999
            else:
                break
        return s.extreme_quantile

    def predict(self, df_test, timestamps=None, batch_size=128):
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first.")

        self.model.eval()
        
        # Generate data
        try:
            tensor_matrices, _ = self.generator.generate(df_test)
        except ValueError as e:
            print(f"Generation error: {e}")
            return {}
            
        if tensor_matrices.nelement() == 0:
            print(f"Test dataframe too small for window {self.generator.w}.")
            return {}
            
        # Get scores using batch processing
        all_scores = self._get_anomaly_scores(tensor_matrices, batch_size=batch_size)
        
        # Time Alignment
        w = self.generator.w
        s = self.generator.step
        
        if timestamps is not None:
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)
                
            valid_indices = range(w, len(df_test) + 1, s)
            min_len = min(len(all_scores), len(valid_indices))
            
            final_scores = all_scores[:min_len]
            final_indices = valid_indices[:min_len]
            
            final_timestamps = []
            for idx in final_indices:
                # Basic protection for index out of bounds
                if idx < len(ts_values):
                     final_timestamps.append(ts_values[idx])
                elif idx == len(ts_values):
                    final_timestamps.append(ts_values[-1])
            
            # Additional trimming if needed
            final_scores = final_scores[:len(final_timestamps)]
            
            return {
                'timestamps': final_timestamps,
                'scores': final_scores
            }
        else:
            return {
                'timestamps': np.arange(len(all_scores)), # Dummy timestamps
                'scores': all_scores
            }

def contribution(self, df_test, df_sistema, batch_size=32):
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first.")
        
        self.model.eval()
        
        # Generate data
        try:
            tensor_matrices, tensor_values = self.generator.generate(df_test)
        except ValueError as e:
            raise ValueError(f"Generation error: {e}")
            
        if tensor_matrices.nelement() == 0:
             raise ValueError("No matrices generated from input dataframe.")

        # Loaders
        # Need both matrices (for error calc) and values (for reconstruction calc)
        loader = DataLoader(TensorDataset(tensor_matrices, tensor_values), batch_size=batch_size, shuffle=False)
        
        n_features = tensor_matrices.shape[2]
        total_error_matrix = torch.zeros(n_features, n_features).to(self.device)
        
        original_vals = []
        reconstructed_vals = []
        
        with torch.no_grad():
            self.model.h_state = None
            self.model.c_state = None
            self.model.history = []
            
            for batch_mat, batch_val in loader:
                inputs = batch_mat.to(self.device)
                vals = batch_val.to(self.device)
                
                recon_matrix, recon_val, _, _ = self.model(inputs)
                
                # Contribution Calculation
                # Matrix Error: (B, N, N)
                # squeeze(1) removes channel dim if present (B, 1, N, N) -> (B, N, N)
                if inputs.dim() == 4:
                    diff = inputs - recon_matrix
                    batch_error = torch.pow(diff, 2).squeeze(1)
                else:
                    batch_error = torch.pow(inputs - recon_matrix, 2)
                    
                total_error_matrix += torch.sum(batch_error, dim=0)
                
                # Reconstruction Storage
                original_vals.extend(vals.cpu().numpy())
                reconstructed_vals.extend(recon_val.cpu().numpy())

        variable_scores = torch.sum(total_error_matrix, dim=1).cpu().numpy()
        variable_names = df_test.columns
        total_period_error = np.sum(variable_scores)
        
        # Safe division
        contrib_pct = (variable_scores / total_period_error * 100) if total_period_error > 0 else np.zeros_like(variable_scores)

        # Create temporary DataFrame with calculated scores
        df_contrib = pd.DataFrame({
            'VARIAVEL': variable_names,
            'score': variable_scores,
            '%': contrib_pct
        })
        
        # Merge with df_sistema to bring the DESC column
        df_contrib = pd.merge(df_contrib, df_sistema[['VARIAVEL', 'DESC']], on='VARIAVEL', how='left')
        # Fill in any nulls if a variable from df_test is not in df_sistema
        df_contrib['DESC'] = df_contrib['DESC'].fillna('Sem descrição')
        
        # Sort from highest contribution to lowest
        df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        df_contrib['cum_perc'] = df_contrib['%'].cumsum()
        # Condition 1: Keep variables until cumulative contribution reaches 80%
        mask_cum_80 = df_contrib['cum_perc'].shift(fill_value=0) < 80
        # Condition 2: Keep variables that contribute at least 5% individually
        mask_min_5 = df_contrib['%'] >= 5
        # Keep variables that satisfy either condition
        df_contrib = df_contrib[mask_cum_80 & mask_min_5].drop(columns=['cum_perc']).reset_index(drop=True)

        # Format for the nested dictionary (keys as strings '0', '1', '2'...)
        df_contrib.index = df_contrib.index.astype(str)
        # Reorganize the column order
        df_contrib = df_contrib[['VARIAVEL', 'DESC', 'score', '%']]
        # Convert to the nested dictionary
        contributions_dict = df_contrib.to_dict()

        # Process Reconstructions
        recon_arr = np.array(reconstructed_vals)

        # Create DataFrame with Original/Reconstructed columns
        recon_data = {}
        for i, col in enumerate(variable_names):
            mean = self.generator.mean[col]
            std = self.generator.std[col]
            
            # Denormalize
            rec_real = (recon_arr[:, i] * std) + mean
            recon_data[f"{col}"] = rec_real
            
        reconstruction_df = pd.DataFrame(recon_data)
        
        return contributions_dict, reconstruction_df
