# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RecurrentDistribution(nn.Module):
    """
    Uma implementação PyTorch da distribuição multivariada com estrutura recorrente.
    Simula q(z_{1:T} | x) onde z_t ~ N(mu(x_t, z_{t-1}), sigma(x_t, z_{t-1})).
    """

    def __init__(self, z_dim, mean_q_mlp, std_q_mlp):
        super(RecurrentDistribution, self).__init__()
        self.z_dim = z_dim
        # No PyTorch, passamos os módulos (camadas lineares) que calculam média e std
        self.mean_q_mlp = mean_q_mlp
        self.std_q_mlp = std_q_mlp

    def reparameterize(self, mu, std, n_samples=1):
        """
        Aplica o truque de reparametrização.
        """
        # Se n_samples > 1, expandimos as dimensões: (Batch, Dim) -> (Batch, n_samples, Dim)
        if n_samples > 1:
            mu = mu.unsqueeze(1).expand(-1, n_samples, -1)
            std = std.unsqueeze(1).expand(-1, n_samples, -1)
            
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_q, n_samples=1):
        """
        Equivalente ao método 'sample' do original.
        
        Args:
            input_q (torch.Tensor): Output da RNN (Batch, Seq_Len, Hidden_Dim)
            n_samples (int): Número de amostras z para gerar por input.
            
        Returns:
            z_seq (Tensor): Sequência de latentes amostrados (Batch, Seq, [n_samples], Z_Dim)
            mu_seq (Tensor): Sequência de médias
            std_seq (Tensor): Sequência de desvios padrão
        """
        batch_size, seq_len, _ = input_q.size()
        device = input_q.device

        z_list = []
        mu_list = []
        std_list = []

        # Estado inicial z_{t-1} = 0
        if n_samples > 1:
            z_prev = torch.zeros(batch_size, n_samples, self.z_dim).to(device)
        else:
            z_prev = torch.zeros(batch_size, self.z_dim).to(device)

        # Loop temporal (substitui o tf.scan)
        for t in range(seq_len):
            # Input atual da RNN: x_encoded_t
            h_t = input_q[:, t, :] # (Batch, Hidden)

            # Preparar input para os MLPs: concatenar h_t e z_{t-1}
            # Se n_samples > 1, precisamos expandir h_t para casar com z_prev
            if n_samples > 1:
                h_t_expanded = h_t.unsqueeze(1).expand(-1, n_samples, -1)
                # Combined shape: (Batch, n_samples, Hidden + Z_dim)
                combined_input = torch.cat([h_t_expanded, z_prev], dim=-1)
            else:
                # Combined shape: (Batch, Hidden + Z_dim)
                combined_input = torch.cat([h_t, z_prev], dim=-1)

            # Calcular parâmetros da distribuição
            mu_q = self.mean_q_mlp(combined_input)
            
            # O código original chama self.std_q_mlp. Assumimos que a ativação Softplus
            # já está incluída dentro desse módulo ou será aplicada aqui.
            # Para garantir fidelidade numérica com OmniAnomaly, aplicamos softplus se não for raw.
            # Aqui assumirei que std_q_mlp retorna o valor raw e aplicamos softplus como no wrapper original.
            std_q = F.softplus(self.std_q_mlp(combined_input)) + 1e-4

            # Amostragem (Reparameterization Trick)
            # z_n = mu + noise * std
            z_n = self.reparameterize(mu_q, std_q, n_samples=1 if n_samples==1 else 0) 
            # Nota: Se n_samples > 1, as dimensões já foram tratadas no 'combined_input'

            # Armazenar resultados
            z_list.append(z_n)
            mu_list.append(mu_q)
            std_list.append(std_q)

            # Atualizar z anterior para o próximo passo
            z_prev = z_n

        # Stack para retornar formato de sequência
        # Shape final: (Batch, Seq, [n_samples], Z_Dim)
        z_seq = torch.stack(z_list, dim=1)
        mu_seq = torch.stack(mu_list, dim=1)
        std_seq = torch.stack(std_list, dim=1)

        return z_seq, mu_seq, std_seq

    def log_prob(self, given_z, input_q):
        """
        Calcula a log-probabilidade de uma sequência 'given_z' dada a distribuição gerada por 'input_q'.
        """
        batch_size, seq_len, _ = input_q.size()
        device = input_q.device
        
        # Detectar se given_z tem dimensão de samples extra
        # given_z shape esperado: (Batch, Seq, Z_dim) ou (Batch, Seq, n_samples, Z_dim)
        n_samples = 1
        if given_z.dim() == 4:
            n_samples = given_z.size(2)

        log_prob_list = []
        
        # Estado inicial z_{t-1} = 0
        if n_samples > 1:
            z_prev = torch.zeros(batch_size, n_samples, self.z_dim).to(device)
        else:
            z_prev = torch.zeros(batch_size, self.z_dim).to(device)

        for t in range(seq_len):
            h_t = input_q[:, t, :]
            z_target = given_z[:, t] # O Z que queremos avaliar a probabilidade

            # Preparar inputs (mesma lógica do forward)
            if n_samples > 1:
                h_t_expanded = h_t.unsqueeze(1).expand(-1, n_samples, -1)
                combined_input = torch.cat([h_t_expanded, z_prev], dim=-1)
            else:
                combined_input = torch.cat([h_t, z_prev], dim=-1)

            mu_q = self.mean_q_mlp(combined_input)
            std_q = F.softplus(self.std_q_mlp(combined_input)) + 1e-4

            # Calcular Log Prob: log N(z_target | mu, std)
            dist = Normal(mu_q, std_q)
            # Soma log_prob nas dimensões de Z_dim
            step_log_prob = dist.log_prob(z_target).sum(dim=-1)
            
            log_prob_list.append(step_log_prob)

            # ATENÇÃO À LÓGICA RECORRENTE:
            # Para calcular q(z_t | z_{t-1}), o "z_prev" deve ser o z_{t-1} do próprio dado (given_z),
            # e não um z amostrado aleatoriamente, pois estamos avaliando uma sequência fixa.
            if t > 0:
                z_prev = given_z[:, t-1] # Usa o z anterior da sequência dada
            else:
                # Para t=0, z_prev continua sendo zero
                pass
                
            # No entanto, a lógica do código original TensorFlow (log_prob_step) usa:
            # input_q = tf.concat([given_n, input_q_n], axis=-1) onde given_n é passado pelo scan.
            # No scan do TF para log_prob, ele itera sobre given_z.
            # O "z_previous" no contexto de log_prob é o elemento anterior da sequência 'given_z'.
            # Meu ajuste "if t > 0: z_prev = given_z[:, t-1]" está correto mas precisa de cuidado no t=0.
            # Vamos corrigir para ficar idêntico ao loop:
            
            z_prev = z_target # Para o PRÓXIMO passo (t+1), o z_prev será o z_target atual (t)

        # Stack log_probs
        # Shape: (Batch, Seq, [n_samples])
        return torch.stack(log_prob_list, dim=1)