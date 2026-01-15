# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Predictor']

class Predictor(object):
    """
    OmniAnomaly predictor (PyTorch Version).

    Args:
        model (OmniAnomaly): The :class:`OmniAnomaly` model instance (nn.Module).
        n_z (int or None): Number of `z` samples to take for each `x`.
            (default 1024)
        batch_size (int): Size of each mini-batch for prediction.
            (default 32)
        last_point_only (bool): Whether to obtain the reconstruction
            probability of only the last point in each window?
            (default :obj:`True`)
        device (str or torch.device): Device to run the inference ('cpu' or 'cuda').
    """

    def __init__(self, model, n_z=1024, batch_size=32, last_point_only=True, device='cpu'):
        self._model = model
        self._n_z = n_z
        self._batch_size = batch_size
        self._last_point_only = last_point_only
        self.device = device

    @property
    def model(self):
        return self._model

    def get_score(self, values):
        """
        Get the `reconstruction probability` of specified KPI observations.

        Args:
            values (np.ndarray): 1-D float32 array, the KPI observations.

        Returns:
            np.ndarray: The `reconstruction probability`.
            np.ndarray: The latent variables `z`.
            float: Mean prediction time per batch.
        """
        # Garante que o modelo está em modo de avaliação
        self._model.eval()
        
        # Validar argumentos
        values = np.asarray(values, dtype=np.float32)
        if len(values.shape) != 2:
            raise ValueError('`values` must be a 2-D array')

        # Configurar janela deslizante (Reutiliza a classe utils original, pois é baseada em numpy)
        sliding_window = BatchSlidingWindow(
            array_size=len(values),
            window_size=self._model.window_length,
            batch_size=self._batch_size,
        )

        collector = []
        collector_z = []
        pred_time = []

        # Iterar sobre os mini-batches
        for b_x, in sliding_window.get_iterator([values]):
            start_iter_time = time.time()

            # Converter para Tensor PyTorch e enviar para o dispositivo (GPU/CPU)
            input_x = torch.from_numpy(b_x).float().to(self.device)

            with torch.no_grad():
                # Chama o método get_score do modelo refatorado anteriormente
                # O método get_score do OmniAnomaly refatorado já retorna numpy arrays
                score, z = self._model.get_score(input_x, last_point_only=self._last_point_only)

            collector.append(score)
            collector_z.append(z)
            
            pred_time.append(time.time() - start_iter_time)

        # Concatenar resultados
        result = np.concatenate(collector, axis=0)
        result_z = np.concatenate(collector_z, axis=0)
        
        return result, result_z, np.mean(pred_time)