#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:08:16 2016

@author: Alban Siffer 
@company: Amossys
@license: GNU GPLv3
"""

from math import log, floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.optimize import minimize
import torch  # Adicionado para compatibilidade

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

def _to_numpy(data):
    """
    Função auxiliar para converter entradas (Listas, Tensores PyTorch, Pandas) para NumPy.
    Isso garante que o algoritmo SPOT funcione independente da origem dos dados.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f'Data format {type(data)} is not supported')

"""
================================= MAIN CLASS ==================================
"""

class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
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

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object (Compatível com PyTorch Tensors)
        """
        self.data = _to_numpy(data)
        init_data_np = _to_numpy(init_data)

        # Lógica para definir init_data baseada no tipo ou valor passado
        if isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (init_data < 1) and (init_data > 0):
            r = int(init_data * self.data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            self.init_data = init_data_np

    def add(self, data):
        """
        This function allows to append data to the already fitted data
        """
        data = _to_numpy(data)
        self.data = np.append(self.data, data)

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)
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
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
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
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

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
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you should initialize before running again')
            return {}

        th = []
        alarm = []
        
        for i in tqdm.tqdm(range(self.data.size)):
            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                if self.data[i] > self.extreme_quantile:
                    if with_alarm:
                        alarm.append(i)
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

    def plot(self, run_results, with_alarm=True):
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))
        return fig


"""
============================ UPPER & LOWER BOUNDS =============================
"""

class biSPOT:
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)
    """

    def __init__(self, q=1e-4):
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0
        nonedict = {'up': None, 'down': None}
        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)
        self.Nt = {'up': 0, 'down': 0}

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s
        return s

    def fit(self, init_data, data):
        self.data = _to_numpy(data)
        init_data_np = _to_numpy(init_data)

        if isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (init_data < 1) and (init_data > 0):
            r = int(init_data * self.data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            self.init_data = init_data_np

    def add(self, data):
        data = _to_numpy(data)
        self.data = np.append(self.data, data)

    def initialize(self, verbose=True):
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold['up'] = S[int(0.98 * n_init)]
        self.init_threshold['down'] = S[int(0.02 * n_init)]

        self.peaks['up'] = self.init_data[self.init_data > self.init_threshold['up']] - self.init_threshold['up']
        self.peaks['down'] = -(self.init_data[self.init_data < self.init_threshold['down']] - self.init_threshold['down'])
        self.Nt['up'] = self.peaks['up'].size
        self.Nt['down'] = self.peaks['down'].size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        l = {'up': None, 'down': None}
        for side in ['up', 'down']:
            g, s, l[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        if verbose:
            print('[done]')

    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
        # Reutiliza a implementação estática de SPOT
        return SPOT._rootsFinder(fun, jac, bounds, npoints, method)

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        return SPOT._log_likelihood(Y, gamma, sigma)

    def _grimshaw(self, side, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()
        def v(s):
            return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = biSPOT._rootsFinder(lambda t: w(self.peaks[side], t),
                                         lambda t: jac_w(self.peaks[side], t),
                                         (a + epsilon, -epsilon),
                                         n_points, 'regular')

        right_zeros = biSPOT._rootsFinder(lambda t: w(self.peaks[side], t),
                                          lambda t: jac_w(self.peaks[side], t),
                                          (b, c),
                                          n_points, 'regular')

        zeros = np.concatenate((left_zeros, right_zeros))
        gamma_best = 0
        sigma_best = Ymean
        ll_best = biSPOT._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = biSPOT._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side, gamma, sigma):
        if side == 'up':
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold['up'] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold['up'] - sigma * log(r)
        elif side == 'down':
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold['down'] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold['down'] + sigma * log(r)
        else:
            print('error : the side is not right')

    def run(self, with_alarm=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run.')
            return {}

        thup = []
        thdown = []
        alarm = []
        
        for i in tqdm.tqdm(range(self.data.size)):
            if self.data[i] > self.extreme_quantile['up']:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks['up'] = np.append(self.peaks['up'], self.data[i] - self.init_threshold['up'])
                    self.Nt['up'] += 1
                    self.n += 1
                    g, s, l = self._grimshaw('up')
                    self.extreme_quantile['up'] = self._quantile('up', g, s)

            elif self.data[i] > self.init_threshold['up']:
                self.peaks['up'] = np.append(self.peaks['up'], self.data[i] - self.init_threshold['up'])
                self.Nt['up'] += 1
                self.n += 1
                g, s, l = self._grimshaw('up')
                self.extreme_quantile['up'] = self._quantile('up', g, s)

            elif self.data[i] < self.extreme_quantile['down']:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks['down'] = np.append(self.peaks['down'], -(self.data[i] - self.init_threshold['down']))
                    self.Nt['down'] += 1
                    self.n += 1
                    g, s, l = self._grimshaw('down')
                    self.extreme_quantile['down'] = self._quantile('down', g, s)

            elif self.data[i] < self.init_threshold['down']:
                self.peaks['down'] = np.append(self.peaks['down'], -(self.data[i] - self.init_threshold['down']))
                self.Nt['down'] += 1
                self.n += 1
                g, s, l = self._grimshaw('down')
                self.extreme_quantile['down'] = self._quantile('down', g, s)
            else:
                self.n += 1

            thup.append(self.extreme_quantile['up'])
            thdown.append(self.extreme_quantile['down'])

        return {'upper_thresholds': thup, 'lower_thresholds': thdown, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        x = range(self.data.size)
        K = run_results.keys()
        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'upper_thresholds' in K:
            thup = run_results['upper_thresholds']
            uth_fig, = plt.plot(x, thup, color=deep_saffron, lw=2, ls='dashed')
            fig.append(uth_fig)

        if 'lower_thresholds' in K:
            thdown = run_results['lower_thresholds']
            lth_fig, = plt.plot(x, thdown, color=deep_saffron, lw=2, ls='dashed')
            fig.append(lth_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))
        return fig


"""
================================= WITH DRIFT ==================================
"""

def backMean(X, d):
    # Garante que operamos sobre numpy para performance no loop
    X = _to_numpy(X)
    M = []
    w = X[:d].sum()
    M.append(w / d)
    for i in range(d, len(X)):
        w = w - X[i - d] + X[i]
        M.append(w / d)
    return np.array(M)


class dSPOT(SPOT):
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)
    Inherits from SPOT to avoid code duplication.
    """
    def __init__(self, q, depth):
        super().__init__(q)
        self.depth = depth

    def initialize(self, verbose=True):
        n_init = self.init_data.size - self.depth
        M = backMean(self.init_data, self.depth)
        T = self.init_data[self.depth:] - M[:-1]

        S = np.sort(T)
        self.init_threshold = S[int(0.98 * n_init)]

        self.peaks = T[T > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')

    def run(self, with_alarm=True):
        if self.n > self.init_data.size:
            print('Warning : algorithm already run.')
            return {}

        W = self.init_data[-self.depth:]
        th = []
        alarm = []

        for i in tqdm.tqdm(range(self.data.size)):
            Mi = W.mean()
            if (self.data[i] - Mi) > self.extreme_quantile:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks = np.append(self.peaks, self.data[i] - Mi - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                    W = np.append(W[1:], self.data[i])

            elif (self.data[i] - Mi) > self.init_threshold:
                self.peaks = np.append(self.peaks, self.data[i] - Mi - self.init_threshold)
                self.Nt += 1
                self.n += 1
                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)
                W = np.append(W[1:], self.data[i])
            else:
                self.n += 1
                W = np.append(W[1:], self.data[i])

            th.append(self.extreme_quantile + Mi)

        return {'thresholds': th, 'alarms': alarm}


"""
=========================== DRIFT & DOUBLE BOUNDS =============================
"""

class bidSPOT(biSPOT):
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper and lower bounds)
    """
    def __init__(self, q=1e-4, depth=10):
        super().__init__(q)
        self.depth = depth

    def initialize(self, verbose=True):
        n_init = self.init_data.size - self.depth
        M = backMean(self.init_data, self.depth)
        T = self.init_data[self.depth:] - M[:-1]

        S = np.sort(T)
        self.init_threshold['up'] = S[int(0.98 * n_init)]
        self.init_threshold['down'] = S[int(0.02 * n_init)]

        self.peaks['up'] = T[T > self.init_threshold['up']] - self.init_threshold['up']
        self.peaks['down'] = -(T[T < self.init_threshold['down']] - self.init_threshold['down'])
        self.Nt['up'] = self.peaks['up'].size
        self.Nt['down'] = self.peaks['down'].size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        l = {'up': None, 'down': None}
        for side in ['up', 'down']:
            g, s, l[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        if verbose:
            print('[done]')

    def run(self, with_alarm=True, plot=True):
        if self.n > self.init_data.size:
            print('Warning : algorithm already run.')
            return {}

        W = self.init_data[-self.depth:]
        thup = []
        thdown = []
        alarm = []

        for i in tqdm.tqdm(range(self.data.size)):
            Mi = W.mean()
            Ni = self.data[i] - Mi
            
            if Ni > self.extreme_quantile['up']:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks['up'] = np.append(self.peaks['up'], Ni - self.init_threshold['up'])
                    self.Nt['up'] += 1
                    self.n += 1
                    g, s, l = self._grimshaw('up')
                    self.extreme_quantile['up'] = self._quantile('up', g, s)
                    W = np.append(W[1:], self.data[i])

            elif Ni > self.init_threshold['up']:
                self.peaks['up'] = np.append(self.peaks['up'], Ni - self.init_threshold['up'])
                self.Nt['up'] += 1
                self.n += 1
                g, s, l = self._grimshaw('up')
                self.extreme_quantile['up'] = self._quantile('up', g, s)
                W = np.append(W[1:], self.data[i])

            elif Ni < self.extreme_quantile['down']:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks['down'] = np.append(self.peaks['down'], -(Ni - self.init_threshold['down']))
                    self.Nt['down'] += 1
                    self.n += 1
                    g, s, l = self._grimshaw('down')
                    self.extreme_quantile['down'] = self._quantile('down', g, s)
                    W = np.append(W[1:], self.data[i])

            elif Ni < self.init_threshold['down']:
                self.peaks['down'] = np.append(self.peaks['down'], -(Ni - self.init_threshold['down']))
                self.Nt['down'] += 1
                self.n += 1
                g, s, l = self._grimshaw('down')
                self.extreme_quantile['down'] = self._quantile('down', g, s)
                W = np.append(W[1:], self.data[i])
            else:
                self.n += 1
                W = np.append(W[1:], self.data[i])

            thup.append(self.extreme_quantile['up'] + Mi)
            thdown.append(self.extreme_quantile['down'] + Mi)

        return {'upper_thresholds': thup, 'lower_thresholds': thdown, 'alarms': alarm}