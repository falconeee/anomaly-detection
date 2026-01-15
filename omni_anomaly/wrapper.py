# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# Nota: A classe 'TfpDistribution' foi removida. 
# No PyTorch, utilizamos diretamente 'torch.distributions.Normal' 
# ou outras distribuições dentro do método 'forward' ou 'get_training_loss' do modelo,
# conforme implementado na refatoração da classe OmniAnomaly.

class SoftplusLinear(nn.Module):
    """
    Camada Linear seguida de ativação Softplus com um epsilon para estabilidade numérica.
    Substitui a função: softplus_std
    """
    def __init__(self, in_features, out_features, epsilon=1e-4):
        super(SoftplusLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.epsilon = epsilon

    def forward(self, x):
        return F.softplus(self.linear(x)) + self.epsilon


class RNNBlock(nn.Module):
    """
    Bloco RNN que inclui a célula recorrente seguida por camadas densas opcionais.
    Substitui a função: rnn
    """
    def __init__(self, input_dim, window_length, rnn_num_hidden, 
                 rnn_cell='GRU', hidden_dense=2, dense_dim=200):
        super(RNNBlock, self).__init__()
        
        self.rnn_num_hidden = rnn_num_hidden
        self.window_length = window_length
        
        # Define a célula RNN
        if rnn_cell == 'GRU':
            self.rnn = nn.GRU(input_dim, rnn_num_hidden, batch_first=True)
        elif rnn_cell == 'LSTM':
            self.rnn = nn.LSTM(input_dim, rnn_num_hidden, batch_first=True)
        elif rnn_cell == 'Basic':
            self.rnn = nn.RNN(input_dim, rnn_num_hidden, batch_first=True)
        else:
            raise ValueError("rnn_cell must be LSTM, GRU or Basic")

        # Define as camadas densas subsequentes (conforme lógica original)
        # O código original aplicava 'hidden_dense' camadas de tamanho 'dense_dim' após a RNN
        layers = []
        input_size = rnn_num_hidden
        
        for i in range(hidden_dense):
            layers.append(nn.Linear(input_size, dense_dim))
            # O código original tf.layers.dense usa ativação linear por padrão se não especificado.
            # Se houvesse ativação, adicionaríamos aqui (ex: nn.ReLU()).
            input_size = dense_dim
            
        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Window, Dim)
        Returns:
            Output tensor processado pela RNN e Dense Layers
        """
        # RNN
        # x shape: (batch, seq_len, input_size)
        # self.rnn retorna: output, h_n
        outputs, _ = self.rnn(x)
        
        # Dense Layers
        # Aplicamos a transformação linear em cada passo de tempo
        outputs = self.dense_layers(outputs)
        
        return outputs


class GaussianParamsNet(nn.Module):
    """
    Módulo auxiliar que encapsula a lógica de gerar média e desvio padrão 
    a partir de uma rede oculta.
    Substitui a função: wrap_params_net
    """
    def __init__(self, hidden_net, input_dim_for_head, output_dim, epsilon=1e-4):
        super(GaussianParamsNet, self).__init__()
        self.hidden_net = hidden_net
        
        # Cabeçalhos para média e desvio padrão
        self.mean_layer = nn.Linear(input_dim_for_head, output_dim)
        self.std_layer = SoftplusLinear(input_dim_for_head, output_dim, epsilon=epsilon)

    def forward(self, x):
        # Passa pela rede oculta (ex: RNNBlock)
        h = self.hidden_net(x)
        
        # Gera parâmetros
        mean = self.mean_layer(h)
        std = self.std_layer(h)
        
        return {
            'mean': mean,
            'std': std
        }

# A função wrap_params_net_srnn original era muito específica para o grafo estático do TF
# e retornava apenas o input processado. No PyTorch, isso é apenas o forward pass 
# da rede oculta, não sendo necessário uma classe wrapper específica além do próprio nn.Module.