import pandas as pd
import numpy as np
import math
import random
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class SPOT:
    """ Algoritmo Streaming Peaks-Over-Threshold (EVT) para limiares dinâmicos. """
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
        else: return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            self.level = 1 - level
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

    def _rootsFinder(fun, jac, bounds, npoints, method):
        from scipy.optimize import minimize
        step = (bounds[1] - bounds[0]) / (npoints + 1)
        if step == 0: bounds, step = (0, 1e-4), 1e-5
        X0 = np.arange(bounds[0] + step, bounds[1], step)
        def objFun(X, f, jac):
            g = 0; j = np.zeros(X.shape); i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j
        opt = minimize(lambda X: objFun(X, fun, jac), X0, method='L-BFGS-B', jac=True, bounds=[bounds] * len(X0))
        return np.unique(opt.x)

    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else: L = n * (1 + math.log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s): return 1 + np.log(s).mean()
        def v(s): return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y; return u(s) * v(s) - 1
        def jac_w(Y, t):
            s = 1 + t * Y; us = u(s); vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us
            
        Ym = self.peaks.min(); YM = self.peaks.max(); Ymean = self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon: epsilon = abs(a) / n_points
        a = a + epsilon; b = 2 * (Ymean - Ym) / (Ymean * Ym); c = 2 * (Ymean - Ym) / (Ym ** 2)
        
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t), lambda t: jac_w(self.peaks, t), (a + epsilon, -epsilon), n_points, 'regular')
        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t), lambda t: jac_w(self.peaks, t), (b, c), n_points, 'regular')
        zeros = np.concatenate((left_zeros, right_zeros))
        
        gamma_best = 0; sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1; sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best: gamma_best = gamma; sigma_best = sigma; ll_best = ll
        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0: return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: return self.init_threshold - sigma * math.log(r)


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        
    def forward(self, w):
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        return self.relu(self.linear3(out))
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        return self.sigmoid(self.linear3(out))
    
class UsadCore(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
  
    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        
        loss1 = 1/n * torch.mean((batch - w1)**2) + (1 - 1/n) * torch.mean((batch - w3)**2)
        loss2 = 1/n * torch.mean((batch - w2)**2) - (1 - 1/n) * torch.mean((batch - w3)**2)
        return loss1, loss2


class USAD:
    """ Wrapper principal para orquestração do modelo USAD. """
    def __init__(self, seq_len=10, stride=1, z_size=64, alpha=0.5, beta=0.5, device=None, seed=42):
        self.seq_len = seq_len
        self.stride = stride
        self.z_size = z_size
        self.alpha = alpha
        self.beta = beta
        self.device = device if device else get_default_device()
        
        self.set_deterministic(seed)
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        self.n_features = None

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

    def _reshape_data(self, data):
        xs = []
        for i in range(0, len(data) - self.seq_len + 1, self.stride):
            xs.append(data[i:(i + self.seq_len)])
        return np.array(xs)

    def fit(self, train_data, epochs=70, batch_size=128, lr=1e-4, gain=1.0, verbose=True):
        """
        Main training orchestration pipeline for USAD.
        Handles data preprocessing, adversarial model training, and statistical threshold calibration.
        """
        self.gain = gain
        
        # Standardize input to a list of DataFrames to support training on multiple disjoint periods
        if isinstance(train_data, pd.DataFrame):
            train_data = [train_data]
            
        # Fit scaler
        if verbose: print("Fitting scaler...")
        # Concatenate all normal dataframes to ensure the scaler captures the true global min/max
        full_train_df = pd.concat(train_data, ignore_index=True)
        self.scaler.fit(full_train_df.values)
        
        # If n_features was not set, infer it dynamically from the dataset
        if self.n_features is None:
            self.n_features = full_train_df.shape[1]
            
        # Prepare data
        if verbose: print("Generating training data...")
        train_sequences = []
        
        for df in train_data:
            # Transform and reshape each continuous period independently to avoid creating false windows across gaps
            df_scaled = self.scaler.transform(df.values)
            seqs = self._reshape_data(df_scaled)
            
            if len(seqs) > 0:
                train_sequences.append(torch.tensor(seqs, dtype=torch.float32))
        
        if not train_sequences:
             raise ValueError("No training data generated. Check seq_len and data length!")

        final_train_data = torch.cat(train_sequences, dim=0)
        
        # DataLoader shuffles the batches.
        train_loader = DataLoader(
            TensorDataset(final_train_data), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Initialize Model
        w_size = self.seq_len * self.n_features
        self.model = UsadCore(w_size, self.z_size).to(self.device)
        
        # Two separate optimizers for the Adversarial Training (AE1 and AE2)
        opt1 = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()), lr=lr)
        opt2 = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()), lr=lr)
        
        # Training Phase
        self.model.train()
        if verbose: print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            n = epoch + 1 # Weight factor for adversarial loss
            total_loss1 = 0
            total_loss2 = 0
            
            for (batch,) in train_loader:
                # USAD requires flattened input: (Batch, Seq * Features)
                inputs = batch.to(self.device).view(batch.size(0), -1)
                
                # == Train AE1 (Reconstruction focus) ==
                loss1, _ = self.model.training_step(inputs, n)
                opt1.zero_grad()
                loss1.backward()
                opt1.step()
                
                # == Train AE2 (Adversarial focus) ==
                _, loss2 = self.model.training_step(inputs, n)
                opt2.zero_grad()
                loss2.backward()
                opt2.step()
                
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                
            epoch_duration = time.time() - epoch_start_time

            if verbose and (epoch == 0 or epoch == epochs - 1 or n % 10 == 0):
                avg_loss1 = total_loss1 / len(train_loader)
                avg_loss2 = total_loss2 / len(train_loader)
                print(f"Epoch {n}/{epochs} | Loss1: {avg_loss1:.4f} | Loss2: {avg_loss2:.4f} | Time: {epoch_duration:.2f}s")

        # Post-Training: Threshold Calibration (POT / SPOT Algorithm)
        if verbose: print("Calculating threshold...")
        # Evaluates the model on the normal training data to establish the baseline error distribution
        train_scores = self._get_anomaly_scores(final_train_data, batch_size=batch_size)
        
        self.threshold = self._pot_eval(train_scores)
        
        if verbose:
            print(f"Base Threshold (POT): {self.threshold:.6f}")
            print(f"Gain: {self.gain}")
            print(f"Final Threshold: {self.threshold * self.gain:.6f}")
            
        # Gain acts as a manual sensitivity multiplier 
        # (e.g., gain=1.2 makes alarms 20% less sensitive/higher threshold)
        self.threshold = self.threshold * self.gain

    def _get_anomaly_scores(self, tensor_data, batch_size=128):
        self.model.eval()
        loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=False)
        scores = []
        
        with torch.no_grad():
            for (batch,) in loader:
                inputs = batch.to(self.device).view(batch.size(0), -1)
                z = self.model.encoder(inputs)
                w1 = self.model.decoder1(z)
                w2 = self.model.decoder2(self.model.encoder(w1))
                
                diff1 = torch.mean((inputs - w1)**2, axis=1)
                diff2 = torch.mean((inputs - w2)**2, axis=1)
                score = self.alpha * diff1 + self.beta * diff2
                scores.extend(score.cpu().numpy())
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
        """
        Inference pipeline for new unseen data using the USAD model.
        Returns a dictionary mapping timestamps to their respective anomaly scores (phi).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first!")

        self.model.eval()
        
        # Transform raw test data into sequences (sliding windows)
        test_scaled = self.scaler.transform(df_test.values)
        test_data = self._reshape_data(test_scaled)
        
        if len(test_data) == 0:
            print(f"Test dataframe too small for window {self.seq_len}.")
            return {}
            
        tensor_test = torch.tensor(test_data, dtype=torch.float32)
        
        # Get the raw anomaly score for each window
        all_scores = self._get_anomaly_scores(tensor_test, batch_size=batch_size)
        
        # Time Alignment
        # The reshape_data uses a sliding window (seq_len) and a stride.
        # The score calculated actually represents the state of the system at the *end* of that window.
        w = self.seq_len
        s = self.stride
        
        if timestamps is not None:
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)
                
            # 'valid_indices' mimics the loop in the generator to find which timestamps 
            # correspond to the end of each generated sequence.
            # No USAD original do exemplo anterior, a iteração era `range(0, len(data) - seq_len + 1, stride)`.
            # O índice correspondente ao final da janela `i` é `i + seq_len - 1`.
            valid_indices = [i + w - 1 for i in range(0, len(df_test) - w + 1, s)]
            
            # Truncate to the smallest length to prevent IndexError in case of minor dimension mismatches
            min_len = min(len(all_scores), len(valid_indices))
            final_scores = all_scores[:min_len]
            final_indices = valid_indices[:min_len]
            
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

    def contribution(self, df_test, df_sistema, timestamps=None, batch_size=32):
        """
        Root Cause Analysis (RCA) pipeline para o USAD.
        Identifica os sensores que causaram a anomalia e retorna a projeção (reconstrução).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first!")
        
        self.model.eval()
        
        # Geração dos dados e janelas
        test_scaled = self.scaler.transform(df_test.values)
        test_data = self._reshape_data(test_scaled)
        
        if len(test_data) == 0:
            raise ValueError("No matrices/windows generated from input dataframe!")

        tensor_test = torch.tensor(test_data, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_test), batch_size=batch_size, shuffle=False)
        
        # Acumuladores de Erro e Reconstrução
        total_error_per_feature = torch.zeros(self.n_features).to(self.device)
        reconstructed_vals = []
        
        with torch.no_grad():
            for (batch,) in loader:
                inputs = batch.to(self.device)
                
                # Flatten para o USAD
                inputs_flat = inputs.view(inputs.size(0), -1)
                
                # Forward Pass
                z = self.model.encoder(inputs_flat)
                w1 = self.model.decoder1(z)
                w2 = self.model.decoder2(self.model.encoder(w1))
                
                # Redimensionar para (Batch, Seq_Len, Features) para separar as variáveis
                w1_3d = w1.view(inputs.size(0), self.seq_len, self.n_features)
                w2_3d = w2.view(inputs.size(0), self.seq_len, self.n_features)
                inputs_3d = inputs.view(inputs.size(0), self.seq_len, self.n_features)
                
                # Cálculo do Erro Ponderado
                diff1 = torch.pow(inputs_3d - w1_3d, 2)
                diff2 = torch.pow(inputs_3d - w2_3d, 2)
                weighted_error = (self.alpha * diff1) + (self.beta * diff2)
                
                # Isolar apenas o ÚLTIMO ponto da janela (instante atual t)
                last_point_error = weighted_error[:, -1, :]
                total_error_per_feature += torch.sum(last_point_error, dim=0)
                
                # Guardar os valores reconstruídos pelo AE1 (Decodificador 1) para o df_projection
                last_point_recon = w1_3d[:, -1, :]
                reconstructed_vals.extend(last_point_recon.cpu().numpy())

        # Score Processing
        variable_scores = total_error_per_feature.cpu().numpy()
        variable_names = df_test.columns
        total_period_error = np.sum(variable_scores)
        if total_period_error == 0:
            total_period_error = 1e-9
        
        contrib_pct = (variable_scores / total_period_error) * 100

        # Montagem do DataFrame bruto
        df_contrib = pd.DataFrame({
            'VARIAVEL': variable_names,
            'score': variable_scores,
            '%': contrib_pct
        })
        
        df_contrib = pd.merge(df_contrib, df_sistema[['VARIAVEL', 'DESC', 'SISTEMA']], on='VARIAVEL', how='left')
        df_contrib['DESC'] = df_contrib['DESC'].fillna('NoDesc')
        df_contrib['SISTEMA'] = df_contrib['SISTEMA'].fillna('NoSystem')
        
        df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)

        # Dynamic Identification with MAD Approach
        df_contrib_backup = df_contrib.copy()
        
        median_score = df_contrib['score'].median()
        mad = (df_contrib['score'] - median_score).abs().median()
        
        k = 1.4826
        mad_threshold = median_score + (k * mad)
        
        df_contrib = df_contrib[df_contrib['score'] > mad_threshold].copy()
        
        # Fallback: Se a anomalia for muito sutil para o limiar MAD, retorna o top 3
        if len(df_contrib) == 0:
            df_contrib = df_contrib_backup.head(3).copy()

        # Recalcular os pesos relativos estritamente no subgrupo isolado
        if df_contrib['score'].sum() > 0:
            df_contrib['%'] = (df_contrib['score'] / df_contrib['score'].sum()) * 100
        else:
            df_contrib['%'] = 0.0

        df_contrib.index = df_contrib.index.astype(str)
        df_contrib = df_contrib[['VARIAVEL', 'DESC', 'SISTEMA', 'score', '%']]
        
        contributions_dict = df_contrib.to_dict()

        # Denormalize reconstructed values
        recon_arr = np.array(reconstructed_vals)
        
        # Como o USAD usa MinMaxScaler global (e não Std/Mean independentes), a inversão é direta
        recon_real = self.scaler.inverse_transform(recon_arr)
        reconstruction_df = pd.DataFrame(recon_real, columns=variable_names)
        
        # Align sliding window predictions back to their exact original timestamps
        if timestamps is not None:
            w = self.seq_len
            s = self.stride
            
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)
                
            # Mapeamento do último índice da janela
            valid_indices = [i + w - 1 for i in range(0, len(df_test) - w + 1, s)]
            min_len = min(len(reconstruction_df), len(valid_indices))
            
            reconstruction_df = reconstruction_df.iloc[:min_len].copy()
            valid_indices = valid_indices[:min_len]
            
            aligned_timestamps = []
            for idx in valid_indices:
                if idx < len(ts_values):
                    aligned_timestamps.append(ts_values[idx])
                else:
                    if len(aligned_timestamps) > 0:
                        aligned_timestamps.append(aligned_timestamps[-1])
            
            reconstruction_df.index = aligned_timestamps
            reconstruction_df.index.name = 'timestamp'
            reconstruction_df.reset_index(inplace=True)
        
        return contributions_dict, reconstruction_df