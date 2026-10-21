import pandas as pd
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# =============================================================================
# 1. SPOT (Teoria de Valores Extremos)
# =============================================================================
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

# =============================================================================
# 2. COMPONENTES NEURAIS DO OMNIANOMALY
# =============================================================================
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(1, dim))

    def h(self, x): return torch.tanh(x)
    def h_prime(self, x): return 1 - torch.tanh(x) ** 2

    def forward(self, z):
        w_dot_u = torch.sum(self.w * self.u)
        m_w_dot_u = -1 + torch.log(1 + torch.exp(w_dot_u))
        u_hat = self.u + (m_w_dot_u - w_dot_u) * self.w / (torch.sum(self.w ** 2) + 1e-7)
        lin = F.linear(z, self.w, self.b) 
        f_z = z + u_hat * self.h(lin)
        psi = self.h_prime(lin) * self.w
        det = 1 + torch.sum(u_hat * psi, dim=-1, keepdim=True)
        log_det = torch.log(torch.abs(det) + 1e-7)
        return f_z, log_det

class FlowSequential(nn.Sequential):
    def forward(self, z):
        log_det_sum = 0
        for modules in self:
            z, log_det = modules(z)
            log_det_sum = log_det_sum + log_det
        return z, log_det_sum

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, rnn_num_layers=1):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.gru = nn.GRU(x_dim, hidden_dim, rnn_num_layers, batch_first=True)
        self.post_rnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1)
        )
        self.z_mean = nn.Linear(hidden_dim + z_dim, z_dim)
        self.z_std = nn.Linear(hidden_dim + z_dim, z_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_seq, _ = self.gru(x)
        h_seq = self.post_rnn(h_seq)
        
        z_samples, mu_list, std_list = [], [], []
        z_prev = torch.zeros(batch_size, self.z_dim).to(x.device)
        
        for t in range(seq_len):
            inp = torch.cat([h_seq[:, t, :], z_prev], dim=1)
            mu = self.z_mean(inp)
            std = F.softplus(self.z_std(inp)) + 1e-4
            z_t = mu + std * torch.randn_like(mu)
            
            z_samples.append(z_t)
            mu_list.append(mu)
            std_list.append(std)
            z_prev = z_t
            
        return torch.stack(z_samples, dim=1), torch.stack(mu_list, dim=1), torch.stack(std_list, dim=1)

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, rnn_num_layers=1):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(z_dim, hidden_dim, rnn_num_layers, batch_first=True)
        self.post_rnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1)
        )
        self.x_mean = nn.Linear(hidden_dim, x_dim)
        self.x_std = nn.Linear(hidden_dim, x_dim)
        
    def forward(self, z):
        h_seq, _ = self.gru(z)
        h_seq = self.post_rnn(h_seq)
        return self.x_mean(h_seq), F.softplus(self.x_std(h_seq)) + 1e-4

class OmniAnomalyModel(nn.Module):
    def __init__(self, x_dim, z_dim=3, hidden_dim=500, window_length=100, nf_layers=20):
        super(OmniAnomalyModel, self).__init__()
        self.window_length = window_length
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim)
        self.flow = FlowSequential(*[PlanarFlow(z_dim) for _ in range(nf_layers)]) if nf_layers > 0 else None
            
    def forward(self, x):
        z_gen, z_mu, z_std = self.encoder(x)
        if self.flow is not None:
            z_fin, log_det_jac = self.flow(z_gen)
        else:
            z_fin, log_det_jac = z_gen, 0
            
        x_rec_mu, x_rec_std = self.decoder(z_fin)
        return {
            'x_rec_mu': x_rec_mu, 'x_rec_std': x_rec_std,
            'z_gen': z_gen, 'z_fin': z_fin, 'z_mu': z_mu, 'z_std': z_std,
            'log_det_jac': log_det_jac
        }

    def loss_function(self, x, output):
        recon_dist = torch.distributions.Normal(output['x_rec_mu'], output['x_rec_std'])
        log_p_x_given_z = recon_dist.log_prob(x).sum(dim=[-1, -2])
        
        q_dist = torch.distributions.Normal(output['z_mu'], output['z_std'])
        log_q_z_gen = q_dist.log_prob(output['z_gen']).sum(dim=[-1, -2])
        log_det = output['log_det_jac'].sum(dim=[-1, -2]) if isinstance(output['log_det_jac'], torch.Tensor) else 0
        log_q_z_fin = log_q_z_gen - log_det
        
        z_t = output['z_fin']
        z_t_minus_1 = torch.cat([torch.zeros(z_t.size(0), 1, z_t.size(2)).to(z_t.device), z_t[:, :-1, :]], dim=1)
        prior_dist = torch.distributions.Normal(z_t_minus_1, torch.ones_like(z_t))
        log_p_z = prior_dist.log_prob(z_t).sum(dim=[-1, -2])
        
        elbo = log_p_x_given_z + log_p_z - log_q_z_fin
        return -elbo.mean()

# =============================================================================
# 3. WRAPPER PRINCIPAL (PIPELINE)
# =============================================================================
class OmniAnomaly:
    def __init__(self, seq_len=100, stride=1, z_dim=3, hidden_dim=500, nf_layers=20, device=None, seed=42):
        self.seq_len = seq_len
        self.stride = stride
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nf_layers = nf_layers
        self.seed = seed
        
        self.set_deterministic(self.seed)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
        windows = [data[i : i + self.seq_len] for i in range(0, len(data) - self.seq_len + 1, self.stride)]
        return np.array(windows) if len(windows) > 0 else np.array([])

    def _get_anomaly_scores(self, data_tensor, batch_size=128):
        """Calcula a Log-Probabilidade Negativa por janela para uso no SPOT."""
        self.model.eval()
        scores = []
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                
                dist = torch.distributions.Normal(output['x_rec_mu'], output['x_rec_std'])
                log_prob = dist.log_prob(inputs) # [batch, seq, dims]
                
                # Anomaly Score = Neg Log Prob do ÚLTIMO passo no tempo
                score = -log_prob[:, -1, :].sum(dim=-1)
                scores.extend(score.cpu().numpy())
                
        return np.array(scores)

    def _pot_eval(self, init_score, q=1e-4, level=0.01):
        """Inicializa o limiar do SPOT."""
        lms = level
        while True:
            try:
                s = SPOT(q)
                s.fit(init_score, init_score)
                s.initialize(level=lms, verbose=False)
                break
            except Exception:
                lms *= 0.999
        return s.extreme_quantile

    def fit(self, df_train, epochs=10, batch_size=128, lr=1e-3, gain=1.0):
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
        
        # 4. Escala e gera janelas ISOLADAMENTE para cada dataframe da lista
        all_windows = []
        for df in df_train:
            vals = df.values if isinstance(df, pd.DataFrame) else df
            data_scaled = self.scaler.transform(vals)
            
            windows = self._reshape_data(data_scaled)
            if len(windows) > 0:
                all_windows.append(windows)
                
        if not all_windows:
            raise ValueError(f"Todos os dataframes fornecidos são menores que o tamanho da janela (seq_len={self.seq_len}). Nenhuma janela foi gerada.")
            
        # Concatena todas as janelas geradas em um único array
        final_windows = np.concatenate(all_windows, axis=0)
        
        # Converte para tensor e cria o DataLoader
        tensor_data = torch.tensor(final_windows, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)
        
        # Init Model
        self.model = OmniAnomalyModel(self.n_features, self.z_dim, self.hidden_dim, self.seq_len, self.nf_layers).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Loop de Treino
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0
            for (inputs,) in train_loader:
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.model.loss_function(inputs, output)
                loss.backward()
                
                # Opcional: Gradient Clipping ajuda a estabilizar os Normalizing Flows
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{epochs}] | Train Loss (ELBO): {avg_loss:.4f}")
        
        # Cálculo Dinâmico do Threshold com SPOT
        print("Calculando limiar dinâmico (SPOT)...")
        train_scores = self._get_anomaly_scores(tensor_data, batch_size)
        
        # Remove NaNs se o modelo estourou gradiente (segurança)
        train_scores = np.nan_to_num(train_scores, nan=np.nanmean(train_scores))
        
        base_threshold = self._pot_eval(train_scores)
        self.threshold = base_threshold * self.gain
        print(f"Limiar de Anomalia Final (SPOT * Gain): {self.threshold:.4f}")

    def predict(self, df_test, timestamps=None, batch_size=128):
        """
        Inference pipeline for new unseen data.
        Returns a dictionary mapping timestamps to their respective anomaly scores (phi).
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado. Execute .fit() primeiro.")

        self.model.eval()
        
        # Transformação dos dados brutos de teste em sequências (sliding windows)
        vals = df_test.values if isinstance(df_test, pd.DataFrame) else df_test
        try:
            data_scaled = self.scaler.transform(vals)
            windows = self._reshape_data(data_scaled)
        except Exception as e:
            print(f"Erro na transformação dos dados: {e}")
            return {}
        
        if len(windows) == 0:
            print(f"Dataframe de teste muito pequeno para a janela de tamanho {self.seq_len}.")
            return {}
            
        tensor_windows = torch.tensor(windows, dtype=torch.float32)
        
        # Obtém o score de anomalia bruto para cada janela
        all_scores = self._get_anomaly_scores(tensor_windows, batch_size=batch_size)
        
        # Alinhamento Temporal (Time Alignment)
        # O _reshape_data usa uma janela deslizante (seq_len) e um stride.
        # O score calculado representa o estado do sistema no *final* daquela janela.
        w = self.seq_len
        s = self.stride
        
        if timestamps is not None:
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)
                
            # valid_indices mimetiza o loop do gerador para encontrar quais timestamps 
            # correspondem ao final de cada sequência gerada.
            valid_indices = [i + w - 1 for i in range(0, len(df_test) - w + 1, s)]
            
            # Trunca para o menor tamanho para evitar IndexError em caso de divergências de dimensão
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
            
            final_scores = abs(final_scores[:len(final_timestamps)])
            
            return {
                'timestamp': final_timestamps,
                'phi': final_scores.tolist() if isinstance(final_scores, np.ndarray) else final_scores
            }
        else:
            return {
                'timestamp': np.arange(len(all_scores)).tolist(), # Timestamps fictícios caso nenhum seja fornecido
                'phi': all_scores.tolist() if isinstance(all_scores, np.ndarray) else all_scores
            }

    def contribution(self, df_anomaly, timestamps=None, df_sistema=None, batch_size=32):
        """
        Análise de Causa Raiz baseada na Log-Likelihood Negativa por feature.
        """
        if self.model is None: raise ValueError("Execute .fit() primeiro.")
        
        vals = df_anomaly.values if isinstance(df_anomaly, pd.DataFrame) else df_anomaly
        data_scaled = self.scaler.transform(vals)
        windows = self._reshape_data(data_scaled)
        
        if len(windows) == 0: raise ValueError("Dados insuficientes.")
            
        loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        
        total_feature_score = torch.zeros(self.n_features).to(self.device)
        reconstructed_vals_last_step = []
        
        self.model.eval()
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                
                # Avalia Log Prob
                dist = torch.distributions.Normal(output['x_rec_mu'], output['x_rec_std'])
                log_prob = dist.log_prob(inputs) # [batch, seq, feature]
                
                # Anomalia é falta de probabilidade (Neg Log Prob)
                batch_feature_scores = -log_prob[:, -1, :] # Apenas o último ponto
                total_feature_score += batch_feature_scores.sum(dim=0)
                
                reconstructed_vals_last_step.extend(output['x_rec_mu'][:, -1, :].cpu().numpy())

        # 1. Pipeline de Causa Raiz (MAD)
        variable_scores = total_feature_score.cpu().numpy()
        # Remove negativos se houver devido a variância do flow (garantia de proporção)
        variable_scores = np.clip(variable_scores, a_min=0, a_max=None) 
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

        # Filtro MAD
        median_score = df_contrib['score'].median()
        mad = (df_contrib['score'] - median_score).abs().median()
        mad_threshold = median_score + (1.4826 * mad)
        
        df_contrib_filtered = df_contrib[df_contrib['score'] > mad_threshold].copy()
        
        if len(df_contrib_filtered) == 0:
            df_contrib_filtered = df_contrib.head(3).copy()

        if df_contrib_filtered['score'].sum() > 0:
            df_contrib_filtered['%'] = (df_contrib_filtered['score'] / df_contrib_filtered['score'].sum()) * 100
        
        contributions_dict = df_contrib_filtered[['VARIAVEL', 'DESC', 'SISTEMA', 'score', '%']].to_dict('records')

        # 2. Reconstrução
        recon_real = self.scaler.inverse_transform(np.array(reconstructed_vals_last_step))
        reconstruction_df = pd.DataFrame(recon_real, columns=self.feature_names)
        
        if timestamps is not None:
            ts_values = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)
            valid_timestamps = ts_values[self.seq_len - 1 :: self.stride]
            min_len = min(len(reconstruction_df), len(valid_timestamps))
            
            reconstruction_df = reconstruction_df.iloc[:min_len].copy()
            reconstruction_df.index = valid_timestamps[:min_len]
            reconstruction_df.index.name = 'timestamp'
            reconstruction_df.reset_index(inplace=True)
            
        return contributions_dict, reconstruction_df