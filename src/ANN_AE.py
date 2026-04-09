import pandas as pd
import numpy as np
import math
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

class SPOT:
    """
    Calcula limiares de anomalia baseados na Teoria de Valores Extremos (EVT).
    Modela as 'caudas' da distribuição de erros de reconstrução usando a
    Distribuição Generalizada de Pareto (GPD).
    """
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
        self.init_data = np.array(init_data) if isinstance(init_data, (list, pd.Series)) else init_data
        self.data = np.array(data) if isinstance(data, (list, pd.Series)) else data

    def initialize(self, level=0.98, verbose=True):
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
            
        g, s, _ = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        
        if verbose:
            print(f'SPOT Extreme quantile (probability = {self.proba}): {self.extreme_quantile:.6f}')

    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method='regular'):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            if step == 0: bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        else:
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)
            
        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            for i, x in enumerate(X):
                fx = f(x)
                g += fx ** 2
                j[i] = 2 * fx * jac(x)
            return g, j
            
        opt = minimize(lambda X: objFun(X, fun, jac), X0, method='L-BFGS-B', jac=True, bounds=[bounds] * len(X0))
        return np.unique(np.round(opt.x, decimals=5))

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            return -n * math.log(sigma) - (1 + (1 / gamma)) * np.log(1 + tau * Y).sum()
        return n * (1 + math.log(Y.mean()))

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s): return 1 + np.log(s).mean()
        def v(s): return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            return u(s) * v(s) - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us, vs = u(s), v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us
            
        Ym, YM, Ymean = self.peaks.min(), self.peaks.max(), self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon: epsilon = abs(a) / n_points
        a += epsilon
        b, c = 2 * (Ymean - Ym) / (Ymean * Ym), 2 * (Ymean - Ym) / (Ym ** 2)
        
        left_zeros = self._rootsFinder(lambda t: w(self.peaks, t), lambda t: jac_w(self.peaks, t), (a + epsilon, -epsilon), n_points)
        right_zeros = self._rootsFinder(lambda t: w(self.peaks, t), lambda t: jac_w(self.peaks, t), (b, c), n_points)
        zeros = np.concatenate((left_zeros, right_zeros))
        
        gamma_best, sigma_best, ll_best = 0, Ymean, self._log_likelihood(self.peaks, 0, Ymean)
        
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = self._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best, sigma_best, ll_best = gamma, sigma, ll
                
        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0: return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        return self.init_threshold - sigma * math.log(r)


class AE_Model(nn.Module):
    def __init__(self, seq_len=10, n_features=3, latent_dim=32):
        super(AE_Model, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.input_dim = seq_len * n_features
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, self.input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        latent = self.encoder(x_flat)
        reconstruction_flat = self.decoder(latent)
        return reconstruction_flat.view(batch_size, self.seq_len, self.n_features)


class ANN_AE:
    def __init__(self, seq_len=10, stride=1, latent_dim=32, device=None, seed=42):
        self.seq_len = seq_len
        self.stride = stride
        self.latent_dim = latent_dim
        self.seed = seed
        
        self.set_deterministic(self.seed)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        self.gain = 1.0
        self.n_features = None
        self.feature_names = None

    def set_deterministic(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def _reshape_data(self, data):
        return np.array([data[i:(i + self.seq_len)] for i in range(0, len(data) - self.seq_len + 1, self.stride)])

    def _get_anomaly_scores(self, data_tensor, batch_size=128):
        """Calcula o erro de reconstrução MAE por janela para uso no SPOT."""
        self.model.eval()
        scores = []
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                reconstructions = self.model(inputs)
                batch_loss = torch.mean(torch.abs(reconstructions - inputs), dim=[1, 2])
                scores.extend(batch_loss.cpu().numpy())
        return np.array(scores)

    def _pot_eval(self, init_score, q=1e-4, level=0.02):
        """Inicializa o limiar do SPOT."""
        lms = level
        while True:
            try:
                s = SPOT(q)
                s.fit(init_score, init_score)
                s.initialize(level=lms, verbose=False)
                break
            except Exception:
                lms *= 0.999 # Diminui o rigor se falhar ao ajustar a curva de Pareto
        return s.extreme_quantile

    def fit(self, df_train, epochs=100, batch_size=128, lr=1e-3, patience=20, val_split=0.1, gain=1.0):
        self.gain = gain
        
        # 1. Padroniza a entrada para ser sempre uma lista
        if isinstance(df_train, pd.DataFrame) or isinstance(df_train, np.ndarray):
            df_train = [df_train]
            
        # 2. Extrai metadados do primeiro dataframe
        first_df = df_train[0]
        self.n_features = first_df.shape[1]
        self.feature_names = first_df.columns.tolist() if isinstance(first_df, pd.DataFrame) else [f"Var_{i}" for i in range(self.n_features)]
        
        # 3. Treina o Scaler globalmente com todos os dados
        if isinstance(first_df, pd.DataFrame):
            full_data = pd.concat(df_train, ignore_index=True).values
        else:
            full_data = np.vstack(df_train)
        self.scaler.fit(full_data)
        
        # 4. Escala e gera janelas isoladamente para cada período
        all_windows = []
        for df in df_train:
            vals = df.values if isinstance(df, pd.DataFrame) else df
            data_scaled = self.scaler.transform(vals)
            windows = self._reshape_data(data_scaled)
            if len(windows) > 0:
                all_windows.append(windows)
                
        if not all_windows:
            raise ValueError("Os dataframes são menores que o seq_len. Nenhuma janela foi gerada.")
            
        # Concatena as janelas finais
        final_windows = np.concatenate(all_windows, axis=0)
        tensor_data = torch.tensor(final_windows, dtype=torch.float32)
        
        # 5. Restante do pipeline permanece inalterado
        split_idx = int((1 - val_split) * len(tensor_data))
        train_loader = DataLoader(TensorDataset(tensor_data[:split_idx]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(tensor_data[split_idx:]), batch_size=batch_size, shuffle=False)
        
        self.model = AE_Model(self.seq_len, self.n_features, self.latent_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0
            for (inputs,) in train_loader:
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                reconstructions = self.model(inputs)
                loss = criterion(reconstructions, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for (inputs,) in val_loader:
                    inputs = inputs.to(self.device)
                    reconstructions = self.model(inputs)
                    loss = criterion(reconstructions, inputs)
                    val_loss += loss.item()
            
            avg_train, avg_val = train_loss / len(train_loader), val_loss / len(val_loader)
            
            if avg_val < best_val_loss:
                best_val_loss, patience_counter, best_model_state = avg_val, 0, self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break
                
        self.model.load_state_dict(best_model_state)
        
        # Cálculo Dinâmico do Threshold com SPOT
        print("Calculando limiar dinâmico (SPOT)...")
        train_scores = self._get_anomaly_scores(tensor_data)
        base_threshold = self._pot_eval(train_scores)
        self.threshold = base_threshold * self.gain
        print(f"Limiar de Anomalia Final (SPOT * Gain): {self.threshold:.6f}")

    def predict(self, df_test, timestamps=None, batch_size=128):
        """
        Inference pipeline for new unseen data.
        Returns a dictionary mapping timestamps to their respective anomaly scores (loss) and classification.
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado. Execute .fit() primeiro.")

        self.model.eval()
        
        # Transformação dos dados de teste
        try:
            data_scaled = self.scaler.transform(df_test.values)
            windows = self._reshape_data(data_scaled)
        except Exception as e:
            print(f"Erro na transformação dos dados: {e}")
            return {}
            
        if len(windows) == 0:
            print(f"Dataframe de teste muito pequeno para a janela de tamanho {self.seq_len}.")
            return {}
            
        # Obtém o score de anomalia (erro de reconstrução absoluto/MSE)
        tensor_windows = torch.tensor(windows, dtype=torch.float32)
        all_scores = self._get_anomaly_scores(tensor_windows, batch_size=batch_size)
        
        # Alinhamento Temporal (Time Alignment)
        # O score calculado representa o estado do sistema no *final* daquela janela.
        w = self.seq_len
        s = self.stride
        
        if timestamps is not None:
            ts_values = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)
                
            # Mapeia os índices correspondentes ao último timestamp de cada janela gerada
            valid_indices = [i + w - 1 for i in range(0, len(df_test) - w + 1, s)]
            
            # Trunca para evitar IndexError em caso de divergências de dimensão
            min_len = min(len(all_scores), len(valid_indices))
            final_scores = all_scores[:min_len]
            final_indices = valid_indices[:min_len]
            
            final_timestamps = []
            for idx in final_indices:
                # Proteção básica para index fora dos limites
                if idx < len(ts_values):
                     final_timestamps.append(ts_values[idx])
                elif idx == len(ts_values):
                    # Se bater exatamente no limite, pega o último timestamp disponível
                    final_timestamps.append(ts_values[-1])
            
            final_scores = final_scores[:len(final_timestamps)]
            
            return {
                'timestamp': final_timestamps,
                'phi': final_scores.tolist()
            }
        else:
            # Timestamps fictícios caso nenhum seja fornecido
            return {
                'timestamp': np.arange(len(all_scores)).tolist(),
                'phi': all_scores.tolist()
            }

    def contribution(self, df_anomaly, timestamps=None, df_sistema=None, batch_size=32):
        """
        Análise de Causa Raiz: Calcula o erro de reconstrução individual por variável usando
        MSE e aplica o método MAD para isolar o sensor culpado da anomalia.
        Retorna o dicionário de contribuições e o DataFrame dos dados reconstruídos desnormalizados.
        """
        if self.model is None: raise ValueError("Execute .fit() primeiro.")
        
        data_scaled = self.scaler.transform(df_anomaly.values)
        windows = self._reshape_data(data_scaled)
        if len(windows) == 0: raise ValueError("Dados insuficientes para formar uma janela.")
            
        loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        
        total_error_val = torch.zeros(self.n_features).to(self.device)
        reconstructed_vals_last_step = []
        
        self.model.eval()
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                recon = self.model(inputs)
                
                # Erro quadrático médio (MSE) por variável ao longo da janela e do batch
                batch_error = torch.pow(inputs - recon, 2)
                total_error_val += torch.sum(batch_error, dim=[0, 1])
                
                # Guarda apenas o último timestep de cada janela para a reconstrução temporal
                reconstructed_vals_last_step.extend(recon[:, -1, :].cpu().numpy())

        # 1. Pipeline de Causa Raiz (MAD)
        variable_scores = total_error_val.cpu().numpy()
        total_period_error = np.sum(variable_scores)
        contrib_pct = (variable_scores / total_period_error * 100) if total_period_error > 0 else np.zeros_like(variable_scores)

        df_contrib = pd.DataFrame({
            'VARIAVEL': self.feature_names,
            'score': variable_scores,
            '%': contrib_pct
        })
        
        if df_sistema is not None:
            df_contrib = pd.merge(df_contrib, df_sistema[['VARIAVEL', 'DESC', 'SISTEMA']], on='VARIAVEL', how='left')
            df_contrib.fillna({'DESC': 'NoDesc', 'SISTEMA': 'NoSystem'}, inplace=True)
        else:
            df_contrib['DESC'] = 'N/A'
            df_contrib['SISTEMA'] = 'N/A'
            
        df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)

        # Filtro MAD (Median Absolute Deviation)
        median_score = df_contrib['score'].median()
        mad = (df_contrib['score'] - median_score).abs().median()
        mad_threshold = median_score + (1.4826 * mad) # 1.4826 aproxima o MAD do desvio padrão
        
        df_contrib_filtered = df_contrib[df_contrib['score'] > mad_threshold].copy()
        
        # Fallback: Se nenhuma variável superar o threshold, mostra as top 3
        if len(df_contrib_filtered) == 0:
            df_contrib_filtered = df_contrib.head(3).copy()

        if df_contrib_filtered['score'].sum() > 0:
            df_contrib_filtered['%'] = (df_contrib_filtered['score'] / df_contrib_filtered['score'].sum()) * 100
        
        contributions_dict = df_contrib_filtered[['VARIAVEL', 'DESC', 'SISTEMA', 'score', '%']].to_dict()

        # 2. Pipeline de Reconstrução Desnormalizada
        recon_arr = np.array(reconstructed_vals_last_step)
        # Inverte o Z-score/MinMax para a grandeza física original
        recon_real = self.scaler.inverse_transform(recon_arr)
        
        reconstruction_df = pd.DataFrame(recon_real, columns=self.feature_names)
        
        # Alinha os timestamps com os dados reconstruídos
        if timestamps is not None:
            ts_values = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)
            valid_timestamps = ts_values[self.seq_len - 1 :: self.stride]
            min_len = min(len(reconstruction_df), len(valid_timestamps))
            
            reconstruction_df = reconstruction_df.iloc[:min_len].copy()
            reconstruction_df.index = valid_timestamps[:min_len]
            reconstruction_df.index.name = 'timestamp'
            reconstruction_df.reset_index(inplace=True)
            
        return contributions_dict, reconstruction_df