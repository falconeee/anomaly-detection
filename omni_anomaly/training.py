# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn
from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Trainer']

class Trainer(object):
    """
    OmniAnomaly trainer (PyTorch Version).
    
    Mantém a fidelidade à lógica original de:
    - Sliding Window batching
    - Gradient Clipping
    - LR Annealing manual
    - Separação Treino/Validação
    """

    def __init__(self, model, n_z=None,
                 optimizer_params=None,
                 max_epoch=256, max_step=None, batch_size=256,
                 valid_batch_size=1024, valid_step_freq=100,
                 initial_lr=0.001, lr_anneal_epochs=10, lr_anneal_factor=0.75,
                 grad_clip_norm=50.0, l2_reg=0.0, device='cpu'):
        
        self._model = model
        self._n_z = n_z
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._valid_step_freq = valid_step_freq
        self._initial_lr = initial_lr
        self._lr_anneal_epochs = lr_anneal_epochs
        self._lr_anneal_factor = lr_anneal_factor
        self._grad_clip_norm = grad_clip_norm
        self.device = device

        # Configurar Otimizador
        # Nota: L2 Regularization no PyTorch é feito via weight_decay no otimizador
        if optimizer_params is None:
            optimizer_params = {}
        
        # Garante que o LR e Weight Decay (L2) estejam configurados
        optimizer_params['lr'] = initial_lr
        if l2_reg > 0:
            optimizer_params['weight_decay'] = l2_reg

        self._optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        
        # Variável para controle de melhor modelo (Early Stopping simples)
        self.best_valid_loss = float('inf')

    @property
    def model(self):
        return self._model

    def _update_learning_rate(self, new_lr):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = new_lr

    def fit(self, values, valid_portion=0.3):
        """
        Train the model with given data.
        """
        # Validar dados
        values = np.asarray(values, dtype=np.float32)
        if len(values.shape) != 2:
            raise ValueError('`values` must be a 2-D array')

        # Split Treino/Validação
        n = int(len(values) * valid_portion)
        train_values, v_x = values[:-n], values[-n:]

        # Iteradores de Janela Deslizante (Utils original)
        train_sliding_window = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.window_length,
            batch_size=self._batch_size,
            shuffle=True,
            ignore_incomplete_batch=True,
        )
        
        valid_sliding_window = BatchSlidingWindow(
            array_size=len(v_x),
            window_size=self.model.window_length,
            batch_size=self._valid_batch_size,
        )

        # Mover modelo para dispositivo
        self._model.to(self.device)
        
        # Variáveis de estado
        global_step = 0
        lr = self._initial_lr
        train_batch_times = []
        valid_batch_times = []
        
        print(f"Iniciando treinamento: {self._max_epoch} épocas.")

        # Loop de Épocas
        for epoch in range(1, self._max_epoch + 1):
            self._model.train()
            train_iterator = train_sliding_window.get_iterator([train_values])
            
            start_epoch_time = time.time()
            
            # Loop de Batches (Treino)
            for (batch_x,) in train_iterator:
                global_step += 1
                batch_start_time = time.time()

                # 1. Preparar dados
                input_x = torch.from_numpy(batch_x).float().to(self.device)

                # 2. Passo de Otimização
                self._optimizer.zero_grad()
                loss = self._model.get_training_loss(input_x)
                loss.backward()

                # 3. Gradient Clipping
                if self._grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip_norm)

                self._optimizer.step()

                # Métricas de tempo
                train_batch_times.append(time.time() - batch_start_time)

                # --- Validação Periódica ---
                if global_step % self._valid_step_freq == 0:
                    self._model.eval()
                    valid_loss_accum = 0.0
                    total_valid_samples = 0
                    
                    # Timing da validação
                    valid_start_time = time.time()
                    
                    v_it = valid_sliding_window.get_iterator([v_x])
                    with torch.no_grad():
                        for (b_v_x,) in v_it:
                            valid_batch_start = time.time()
                            
                            valid_input = torch.from_numpy(b_v_x).float().to(self.device)
                            v_loss = self._model.get_training_loss(valid_input)
                            
                            batch_size_v = b_v_x.shape[0]
                            valid_loss_accum += v_loss.item() * batch_size_v
                            total_valid_samples += batch_size_v
                            
                            valid_batch_times.append(time.time() - valid_batch_start)

                    avg_valid_loss = valid_loss_accum / total_valid_samples
                    valid_duration = time.time() - valid_start_time
                    
                    # Atualizar melhor loss
                    if avg_valid_loss < self.best_valid_loss:
                        self.best_valid_loss = avg_valid_loss
                        # Aqui você poderia salvar o checkpoint do modelo

                    print(f"Step {global_step} | Epoch {epoch} | "
                          f"Train Loss: {loss.item():.4f} | Valid Loss: {avg_valid_loss:.4f} | "
                          f"Valid Time: {valid_duration:.2f}s")
                    
                    self._model.train() # Voltar para modo treino

                # Verificar max_step
                if self._max_step is not None and global_step >= self._max_step:
                    break

            # --- LR Annealing (Ao final da época) ---
            if self._lr_anneal_epochs and epoch % self._lr_anneal_epochs == 0:
                lr *= self._lr_anneal_factor
                self._update_learning_rate(lr)
                print(f"Learning rate decreased to {lr:.6f}")

            if self._max_step is not None and global_step >= self._max_step:
                print("Max steps reached.")
                break

        return {
            'best_valid_loss': self.best_valid_loss,
            'train_time': np.mean(train_batch_times),
            'valid_time': np.mean(valid_batch_times),
        }