import pandas as pd
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# ==========================================
# 1. CAMADA ESTATÍSTICA (SPOT)
# ==========================================
class SPOT:
    """
    Calcula limiares de anomalia baseados na Teoria de Valores Extremos (EVT).
    Modela as 'caudas' da distribuição de erros de reconstrução.
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


# ==========================================
# 2. CAMADA DE DEEP LEARNING (Módulos TranAD)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TranAD_Net(nn.Module):
    def __init__(self, feats, window_size=10):
        super(TranAD_Net, self).__init__()
        self.n_feats = feats
        self.n_window = window_size
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)
        
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)
        
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)
        
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Fase 1: Sem score de foco
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        
        # Fase 2: Com self-conditioning (erro da Fase 1 atua como foco)
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        
        return x1, x2


# ==========================================
# 3. CAMADA ORQUESTRADORA (Pipeline)
# ==========================================
class TranAD:
    def __init__(self, seq_len=10, stride=1, device=None, seed=42):
        self.seq_len = seq_len
        self.stride = stride
        self.seed = seed
        
        self.set_deterministic(self.seed)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.thresholds = None
        self.threshold = 1.0 # Backward compatibility para scripts externos (o phi agora é relativo, limiar passa a ser 1.0)
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
        """Calcula o score de anomalia do TranAD (combinação da Fase 1 e Fase 2)."""
        self.model.eval()
        scores = []
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                window_input = inputs.permute(1, 0, 2) # [Seq, Batch, Feats]
                
                # O TranAD opera na janela inteira
                x1, x2 = self.model(window_input, window_input)
                
                # Metodologia TranAD: Eq 13 (0.5 * L1 + 0.5 * L2) 
                # Norma L2 (RMSE no eixo temporal da janela) por feature
                loss1 = torch.sqrt(torch.mean((x1 - window_input) ** 2, dim=0)) # [Batch, Feats]
                loss2 = torch.sqrt(torch.mean((x2 - window_input) ** 2, dim=0)) # [Batch, Feats]
                batch_loss = 0.5 * loss1 + 0.5 * loss2 # [Batch, Feats]
                
                scores.extend(batch_loss.cpu().numpy())
                
        return np.array(scores)

    def _pot_eval(self, init_score, q=1e-4, level=0.02):
        """Inicializa o limiar dinâmico do SPOT estático."""
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

    def fit(self, df_train, epochs=100, batch_size=128, lr=1e-3, patience=20, val_split=0.1, gain=1.0, verbose=False):
        self.gain = gain
        
        # 1. Padroniza a entrada
        if isinstance(df_train, pd.DataFrame) or isinstance(df_train, np.ndarray):
            df_train = [df_train]
            
        first_df = df_train[0]
        self.n_features = first_df.shape[1]
        self.feature_names = first_df.columns.tolist() if isinstance(first_df, pd.DataFrame) else [f"Var_{i}" for i in range(self.n_features)]
        
        # 2. Treina o Scaler
        full_data = pd.concat(df_train, ignore_index=True).values if isinstance(first_df, pd.DataFrame) else np.vstack(df_train)
        self.scaler.fit(full_data)
        
        # 3. Escala e gera janelas
        all_windows = []
        for df in df_train:
            vals = df.values if isinstance(df, pd.DataFrame) else df
            data_scaled = self.scaler.transform(vals)
            windows = self._reshape_data(data_scaled)
            if len(windows) > 0:
                all_windows.append(windows)
                
        if not all_windows:
            raise ValueError("Os dataframes são menores que o seq_len.")
            
        final_windows = np.concatenate(all_windows, axis=0)
        # Atenção: Usando float32 para evitar erro de mat1/mat2 (Double vs Float)
        tensor_data = torch.tensor(final_windows, dtype=torch.float32) 
        
        # 4. DataLoaders
        split_idx = int((1 - val_split) * len(tensor_data))
        train_loader = DataLoader(TensorDataset(tensor_data[:split_idx]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(tensor_data[split_idx:]), batch_size=batch_size, shuffle=False)
        
        self.model = TranAD_Net(self.n_features, self.seq_len).to(self.device)
        criterion = nn.MSELoss(reduction='none')
        
        # TranAD recomenda AdamW
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 5. Loop de Treinamento
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0
            n = epoch # Fator de balanceamento do TranAD
            
            for (inputs,) in train_loader:
                inputs = inputs.to(self.device)
                window = inputs.permute(1, 0, 2)
                
                optimizer.zero_grad()
                z1, z2 = self.model(window, window)
                
                # Perda calculada segundo a formulação original do TranAD
                loss1 = criterion(z1, window)
                loss2 = criterion(z2, window)
                loss = (1 / n) * loss1 + (1 - 1/n) * loss2
                loss = torch.mean(loss)
                
                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss += loss.item()
                
            scheduler.step()
                
            # Validação
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for (inputs,) in val_loader:
                    inputs = inputs.to(self.device)
                    window = inputs.permute(1, 0, 2)
                    z1, z2 = self.model(window, window)
                    
                    l1 = criterion(z1, window)
                    l2 = criterion(z2, window)
                    loss = (1 / n) * l1 + (1 - 1/n) * l2
                    val_loss += torch.mean(loss).item()
            
            avg_train, avg_val = train_loss / len(train_loader), val_loss / len(val_loader)
            
            if avg_val < best_val_loss:
                best_val_loss, patience_counter, best_model_state = avg_val, 0, self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if (epoch % 5 == 0 or epoch == 1) and verbose:
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
                
            if patience_counter >= patience:
                if verbose: print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break
                
        self.model.load_state_dict(best_model_state)
        
        # 6. Cálculo Dinâmico do Threshold com SPOT
        if verbose: print("Calculando limiar dinâmico (SPOT) por variável...")
        train_scores = self._get_anomaly_scores(tensor_data, batch_size=batch_size) # Shape: [N_windows, Feats]
        self.thresholds = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            try:
                base_threshold = self._pot_eval(train_scores[:, i])
                self.thresholds[i] = base_threshold * self.gain
            except Exception as e:
                # Fallback caso o SPOT falhe para uma variável perfeitamente constante
                self.thresholds[i] = np.percentile(train_scores[:, i], 99) * self.gain
                
        if verbose: print(f"Limiares de Anomalia Calculados para {self.n_features} variáveis.")

    def predict(self, df_test, timestamps=None, batch_size=128):
        """Inference pipeline for new unseen data."""
        if self.model is None:
            raise ValueError("O modelo não foi treinado. Execute .fit() primeiro.")

        try:
            data_scaled = self.scaler.transform(df_test.values)
            windows = self._reshape_data(data_scaled)
        except Exception as e:
            print(f"Erro na transformação dos dados: {e}")
            return {}
            
        if len(windows) == 0:
            print(f"Dataframe de teste muito pequeno para a janela de tamanho {self.seq_len}.")
            return {}
            
        tensor_windows = torch.tensor(windows, dtype=torch.float32)
        all_scores = self._get_anomaly_scores(tensor_windows, batch_size=batch_size)
        
        # Phi é o valor máximo relativo ao threshold da variável
        relative_scores = all_scores / (self.thresholds + 1e-8)
        aggregated_phi = np.max(relative_scores, axis=1)
        
        w = self.seq_len
        s = self.stride
        
        if timestamps is not None:
            ts_values = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)
            valid_indices = [i + w - 1 for i in range(0, len(df_test) - w + 1, s)]
            
            min_len = min(len(aggregated_phi), len(valid_indices))
            final_scores = aggregated_phi[:min_len]
            final_indices = valid_indices[:min_len]
            
            final_timestamps = [ts_values[idx] for idx in final_indices if idx < len(ts_values)]
            final_scores = final_scores[:len(final_timestamps)]
            
            return {
                'timestamp': final_timestamps,
                'phi': final_scores.tolist()
            }
        else:
            return {
                'timestamp': np.arange(len(aggregated_phi)).tolist(),
                'phi': aggregated_phi.tolist()
            }

    def contribution(self, df_anomaly, timestamps=None, df_sistema=None, batch_size=32):
        """Análise de Causa Raiz usando as características do TranAD."""
        if self.model is None: raise ValueError("Execute .fit() primeiro.")
        
        data_scaled = self.scaler.transform(df_anomaly.values)
        windows = self._reshape_data(data_scaled)
        if len(windows) == 0: raise ValueError("Dados insuficientes para formar uma janela.")
            
        loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        
        total_relative_val = np.zeros(self.n_features)
        total_excess_val = np.zeros(self.n_features)
        reconstructed_vals_last_step = []
        
        self.model.eval()
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                window_input = inputs.permute(1, 0, 2)
                
                x1, x2 = self.model(window_input, window_input)
                
                # Erro por variável com norma L2 ao longo da janela
                batch_error_1 = torch.sqrt(torch.mean((x1 - window_input) ** 2, dim=0)) # [Batch, Feats]
                batch_error_2 = torch.sqrt(torch.mean((x2 - window_input) ** 2, dim=0)) # [Batch, Feats]
                batch_error = 0.5 * batch_error_1 + 0.5 * batch_error_2 # [Batch, Feats]
                
                batch_error_np = batch_error.cpu().numpy()
                
                # Cálculo do RCA usando threshold da respectiva feature
                rel_scores = batch_error_np / (self.thresholds + 1e-8)
                exc_scores = np.maximum(0, batch_error_np - self.thresholds)
                
                total_relative_val += np.sum(rel_scores, axis=0)
                total_excess_val += np.sum(exc_scores, axis=0)
                
                # Guarda a reconstrução final da Fase 2 (x2) apenas para o último ponto da janela
                reconstructed_vals_last_step.extend(x2[-1, :, :].cpu().numpy())

        # 1. Pipeline de Causa Raiz (RCA com SPOT)
        total_excess_error = np.sum(total_excess_val)
        
        if total_excess_error > 0:
            contrib_pct = (total_excess_val / total_excess_error * 100)
        else:
            # Fallback caso a anomalia seja muito sutil
            total_rel_error = np.sum(total_relative_val)
            contrib_pct = (total_relative_val / total_rel_error * 100) if total_rel_error > 0 else np.zeros_like(total_relative_val)

        df_contrib = pd.DataFrame({
            'VARIAVEL': self.feature_names,
            'score': total_relative_val,
            '%': contrib_pct
        })
        
        if df_sistema is not None:
            df_contrib = pd.merge(df_contrib, df_sistema[['VARIAVEL', 'DESC', 'SISTEMA']], on='VARIAVEL', how='left')
            df_contrib.fillna({'DESC': 'NoDesc', 'SISTEMA': 'NoSystem'}, inplace=True)
        else:
            df_contrib['DESC'] = 'N/A'
            df_contrib['SISTEMA'] = 'N/A'
            
        df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)

        median_score = df_contrib['score'].median()
        mad = (df_contrib['score'] - median_score).abs().median()
        mad_threshold = median_score + (1.4826 * mad) 
        
        df_contrib_filtered = df_contrib[df_contrib['score'] > mad_threshold].copy()
        
        if len(df_contrib_filtered) == 0:
            df_contrib_filtered = df_contrib.head(3).copy()

        if df_contrib_filtered['score'].sum() > 0:
            df_contrib_filtered['%'] = (df_contrib_filtered['score'] / df_contrib_filtered['score'].sum()) * 100
        
        contributions_dict = df_contrib_filtered[['VARIAVEL', 'DESC', 'SISTEMA', 'score', '%']].to_dict()

        # 2. Pipeline de Reconstrução Desnormalizada
        recon_arr = np.array(reconstructed_vals_last_step)
        recon_real = self.scaler.inverse_transform(recon_arr)
        
        # Cria um DataFrame vazio (NaNs) do mesmo tamanho que a entrada df_anomaly
        reconstruction_df = pd.DataFrame(np.nan, index=np.arange(len(df_anomaly)), columns=self.feature_names)
        
        # Preenche com os valores preditos apenas nos índices correspondentes à saída da janela
        w = self.seq_len
        s = self.stride
        valid_indices = [i + w - 1 for i in range(0, len(df_anomaly) - w + 1, s)]
        
        min_len = min(len(recon_real), len(valid_indices))
        reconstruction_df.iloc[valid_indices[:min_len]] = recon_real[:min_len]
        
        if timestamps is not None:
            ts_values = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)
            reconstruction_df.index = ts_values
            reconstruction_df.index.name = 'timestamp'
            reconstruction_df.reset_index(inplace=True)
            
        return contributions_dict, reconstruction_df