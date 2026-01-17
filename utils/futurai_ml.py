import numpy as np
import pandas as pd
from scipy.stats import f, norm
from scipy.stats.distributions import chi2
import math
from typing import Dict
import json


class FuturaiML:
    """
    Classe para geração do modelo futurai
    """

    def __init__(self, nc: int = 0, gain: int = 1):
        self.gain = gain
        self.nc = nc

    def fit(self, x_train: pd.DataFrame):
        """Função para treinamento do modelo
        :param - data - dataframe com a dataset de treinamento
        :return none
        """

        # Faz o scalling da base
        mean_train = x_train.mean()
        std_train = x_train.std()
        
        df = (x_train - mean_train) / std_train
        self.media = mean_train
        self.std = std_train

        # Aplicação do PCA para redução da quantidade de features do dataset
        linhas_colunas = df.shape

        # Matriz covariança dos dados
        df_array = np.array(df.T)
        matrix_cov = np.cov(df_array)

        # SVD para decomposiação da matriz covariança
        coeff, s, _ = np.linalg.svd(matrix_cov)
        coeff = pd.DataFrame(coeff)

        # Metodo VRE - Calculo das componentes principais
        if self.nc == 0:

            eps_pca = np.eye(linhas_colunas[1])
            vre = []

            for j in range(linhas_colunas[1]):

                # Calculo da C1
                residual = coeff.iloc[:, j : linhas_colunas[1]]
                val3_c1 = residual.dot(residual.T)

                val_ui = []
                for i in range(linhas_colunas[1]):
                    eps_aux = eps_pca[:, i]
                    eps_til = val3_c1.dot(eps_aux.T)
                    aux = (eps_til.T.dot(matrix_cov).dot(eps_til)) / (
                        eps_til.T.dot(eps_til) ** 2
                    )

                    val_ui.append(aux)

                vre.append(sum(val_ui) / (eps_aux.T.dot(matrix_cov).dot(eps_aux)))

            self.nc = vre.index(min(vre))

        # Calculo do PCA
        componentes_principais = coeff.iloc[:, 0 : self.nc]
        residual = coeff.iloc[:, self.nc : linhas_colunas[1]]
        aux_s = np.diag(s)
        df_s = pd.DataFrame(aux_s)
        val1 = df_s.iloc[: self.nc, : self.nc]
        val2_d = componentes_principais.dot(np.linalg.inv(val1)).dot(
            componentes_principais.T
        )
        val3_c1 = residual.dot(residual.T)
        self.c2 = componentes_principais.dot(componentes_principais.T)

        # Componentes principais
        coeff = coeff.iloc[:, 0 : self.nc]
        coeff = np.array(coeff)
        df = np.array(df)
        principal_components = df.dot(coeff)
        principal_components = pd.DataFrame(principal_components)

        self.val_d = val2_d
        self.val_c1 = val3_c1

        # Gera calculo dos limiares
        base_dados = principal_components
        a = self.nc
        ds = s

        alfa = 0.99
        n = base_dados.shape
        n = n[0]

        # Limiar da t2
        t2_lim = (a * (n - 1) * (n + 1) / (n * (n - a))) * f.ppf(alfa, a, n - a)

        # Limiar da Q
        teta1 = sum(ds[a:])
        teta2 = sum(ds[a:] ** 2)
        teta3 = sum(ds[a:] ** 3)

        h0 = 1 - (2 * teta1 * teta3) / (3 * (teta2 ** 2))
        ca = norm.ppf(alfa, 0, 1)
        q_lim = teta1 * (
            (h0 * ca * (math.sqrt(2 * teta2)) / teta1)
            + 1
            + (teta2 * h0 * (h0 - 1)) / (teta1 ** 2)
        ) ** (1 / h0)

        # Limiar phi
        gphi = ((a / t2_lim ** 2) + (teta2 / q_lim ** 2)) / (
            (a / t2_lim) + (teta1 / q_lim)
        )
        hphi = ((a / t2_lim) + (teta1 / q_lim)) ** 2 / (
            (a / t2_lim ** 2) + (teta2 / q_lim ** 2)
        )

        phi_lim = gphi * chi2.ppf(alfa, hphi)

        self.t2_lim = t2_lim
        self.q_lim = q_lim
        self.phi_lim = phi_lim * self.gain

    def predict(self, x_test, eixo_x, points=2) -> Dict:
        """Realiza a predição da base de dados
        :param - x_test: Base de dados para Predição
        :return - phi_matrix: uma matriz"""

        base = (x_test - self.media) / self.std

        abase = np.array(base)
        aval_d = self.val_d
        aval_c1 = self.val_c1
        t2_lim = self.t2_lim
        q_lim = self.q_lim

        # Estatistica Phi
        phi = []
        phi_matrix = (aval_d / t2_lim) + (aval_c1 / q_lim)

        for i in range(len(base)):
            phi.append(float((abase[i, :].T).dot(phi_matrix).dot(abase[i, :])))

        # Filtro
        dataset = pd.DataFrame([list(eixo_x), phi], index=["TIMESTAMP", "PHI"]).T
        dataset["TIMESTAMP"] = pd.to_datetime(
            dataset["TIMESTAMP"], format="%Y/%m/%d %H:%M:%S"
        )
        df_aux = dataset.copy()

        df_aux["status"] = 1
        ############################   Subida  ########################################
        ## Essa parte do código serve para pegar picos onde o motor volta a "funcionar"
        ## por menos de uma hora, ou seja, ele estava desligado, deu um pique de menos
        ## de uma hora e voltou a ficar desligado
        ###############################################################################
        data_aux = df_aux["TIMESTAMP"].min()  # Primeira data do dataframe
        while True:

            # Pega a data da primeira amostra com o valor abaixo do limite
            df_amostra = df_aux[
                (df_aux["PHI"] > self.phi_lim) & (df_aux["TIMESTAMP"] >= data_aux)
            ]

            if not df_amostra.empty:
                data_min = df_amostra["TIMESTAMP"].min()
            else:
                break

            # Pega a primeira data da amostra acima do valor limite depois da amostra acima
            df_amostra = df_aux[
                (df_aux["PHI"] <= self.phi_lim) & (df_aux["TIMESTAMP"] > data_min)
            ]

            if not df_amostra.empty:
                data_aux = df_amostra["TIMESTAMP"].min()

                mask = (df_aux["TIMESTAMP"] >= data_min) & (
                    df_aux["TIMESTAMP"] < data_aux
                )

                df_amostra = df_aux.loc[mask]

                if len(df_amostra) <= points:
                    df_aux["status"].loc[mask] = 0

            else:
                data_aux = df_aux["TIMESTAMP"].max()

                mask = (df_aux["TIMESTAMP"] >= data_min) & (
                    df_aux["TIMESTAMP"] <= data_aux
                )

                df_amostra = df_aux.loc[mask]

                if len(df_amostra) <= points:
                    df_aux["status"].loc[mask] = 0

                break

        dataset.drop(df_aux[df_aux["status"] == 0].index, inplace=True)

        phi = list(dataset["PHI"])
        eixo_x = list(dataset["TIMESTAMP"].dt.strftime("%Y-%m-%d %X"))

        predicao = {"matrix": phi_matrix, "phi": phi, "timestamp": eixo_x}

        return predicao

    def contribuition(self, df, phi, df_sistema, eixo_x, eixo_x_proj=None):
        """Função para gerar o grafico os de Contribuição e
        também lista das varaiveis que mais influênciaram"""

        try:

            df = (df - self.media) / self.std
            linhas_colunas = df.shape

            seq_aux = list(range(len(df)))

            n_fast_list = pd.DataFrame([seq_aux])
            n_fast_list = n_fast_list.T
            n_fast_list.columns = ["SEQ"]
            rci = []

            # Geração da matriz de contribuição
            for x in list(range(linhas_colunas[1])):

                # Definindo epsilon - só entram as variáveis em falta
                eps = np.eye(linhas_colunas[1])
                eps = eps[:, x]
                eps = pd.DataFrame(eps)

                # Definindo Trci - diagonal matrix which the elements are one for the faulty variables
                k = [0] * linhas_colunas[1]
                k[x] = 1
                trci = np.diag(k)

                # equação 13 - reconstrução das variáveis em falta
                termo_1 = -np.linalg.inv(eps.T.dot(phi).dot(eps))
                termo_2 = eps.T
                termo_3 = phi
                termo_4 = pd.DataFrame(np.eye(linhas_colunas[1]) - trci)
                termo_5 = df.T

                n_fast = termo_1.dot(termo_2).dot(termo_3).dot(termo_4).dot(termo_5)
                n_fast = n_fast.T
                n_fast = pd.DataFrame(n_fast)

                # Equação 14 do RCI
                termo_3 = (pd.DataFrame(df.iloc[:, x]).values) - n_fast.values
                termo_1 = termo_3.T
                termo_2 = eps.T.dot(phi).dot(eps)
                termo_2 = float(termo_2.values)

                # RCI contem o score de importancia de cada variavel no momento da falha
                rci.append(float((termo_1 * termo_2).dot(termo_3)))

                n_fast = pd.DataFrame(n_fast)
                n_fast_list = pd.concat([n_fast_list, n_fast], axis=1)

            n_fast_list = n_fast_list.drop(["SEQ"], axis=1)
            n_fast_list.columns = df.columns

            termo_1 = df - n_fast_list
            termo_2 = eps.T.dot(phi).dot(eps)
            termo_3 = termo_2 ** 0.5

            circi = termo_1 * float(termo_3.values)
            circi = circi ** 2

            df_sistema["score"] = 0

            # Monta um dataframe de forma decrescente das varaiveis conforme seu score
            df_rci = pd.DataFrame({"score": rci, "variavel": df.columns})
            df_rci = df_rci.sort_values(by="score", ascending=False)

            for _, row in df_rci.iterrows():
                tag = row["variavel"]
                val = row["score"]
                idx = df_sistema[df_sistema["VARIAVEL"] == tag].index

                df_sistema.loc[idx, "score"] = val

            df_score_dec = df_sistema.sort_values(by="score", ascending=False)


            # Recria a phi tirando um a um as varaiveis que mais influenciam até o phi ficar abaixo do limiar
            val_contr = []
            qtd_aux = 1

            for i, row in df_rci.iterrows():

                val_contr.append(i)

                eps = np.eye(linhas_colunas[1])
                eps = eps[:, val_contr]
                eps = pd.DataFrame(eps)

                # Definindo Trci - diagonal matrix which the elements are one for the faulty variables
                k = [0] * linhas_colunas[1]
                for x in val_contr:
                    k[x] = 1
                trci = np.diag(k)

                # equação 13 - reconstrução das variáveis em falta
                termo_1 = -np.linalg.inv(eps.T.dot(phi).dot(eps))
                termo_2 = eps.T
                termo_3 = phi
                termo_4 = pd.DataFrame(np.eye(linhas_colunas[1]) - trci)
                termo_5 = df.T

                n_fast = termo_1.dot(termo_2).dot(termo_3).dot(termo_4).dot(termo_5)
                n_fast = n_fast.T

                phiast = []
                for x in list(range(linhas_colunas[0])):

                    termo_1 = df.iloc[x, :].values
                    termo_2 = np.eye(linhas_colunas[1]) - trci
                    termo_3 = phi
                    termo_4 = termo_2
                    termo_5 = df.iloc[x, :].T.values
                    termo_6 = n_fast[x, :]
                    termo_7 = eps.T.dot(phi).dot(eps)
                    termo_8 = termo_6.T

                    phiast.append(
                        float(
                            (
                                termo_1.dot(termo_2)
                                .dot(termo_3)
                                .dot(termo_4)
                                .dot(termo_5)
                            )
                            - (termo_6.dot(termo_7).dot(termo_8))
                        )
                    )
                if max(phiast) < self.phi_lim and qtd_aux >= 3:
                    break

                # Quantidade de varaiveis que mais influenciaram
                qtd_aux = qtd_aux + 1


            # Separa as varaiveis que mais influenciam das restantes
            df_score_prin = df_score_dec.iloc[0:qtd_aux]

            df_score_prin.reset_index(inplace=True, drop=True)

            ##### Gera dataframe com a projeção das variáveis PRINCIPAIS #####
            df_projection_prin_vars = pd.DataFrame(n_fast)
            df_projection_prin_vars.columns = list(df_score_prin["VARIAVEL"])
            df_projection_prin_vars = (df_projection_prin_vars*self.std) + self.media
            df_projection_prin_vars = df_projection_prin_vars[list(df_score_prin["VARIAVEL"])]
            # Resample Dataframe
            if eixo_x_proj is not None:
                df_projection_prin_vars['timestamp'] = list(eixo_x_proj)
                df_projection_prin_vars = df_projection_prin_vars.set_index('timestamp')
                df_projection_prin_vars = df_projection_prin_vars.resample('1T').asfreq()
                df_projection_prin_vars.reset_index(inplace=True)
                df_projection_prin_vars.drop('timestamp', inplace=True, axis=1)

            ##### Gera dataframe com a projeção de todas as VARIAVEIS #####
            df_projection_full = pd.DataFrame(n_fast_list)
            df_projection_full = (df_projection_full*self.std) + self.media
            # Resample Dataframe
            if eixo_x_proj is not None:
                df_projection_full['timestamp'] = list(eixo_x_proj)
                df_projection_full = df_projection_full.set_index('timestamp')
                df_projection_full = df_projection_full.resample('1T').asfreq()
                df_projection_full.reset_index(inplace=True)
                df_projection_full.drop('timestamp', inplace=True, axis=1)

            df_score_res = df_score_dec.iloc[qtd_aux:]

            ######## Geração do grafico hierarquico conforme local ########
            soma = df_score_prin["score"].sum()
            df_score_prin["%"] = df_score_prin.score.apply(
                lambda x: round((x / soma * 100), 5)
            )

            soma = df_score_dec["score"].sum()
            df_score_dec["%"] = df_score_dec.score.apply(
                lambda x: round((x / soma * 100), 5)
            )

            # Geração do grafico de contribuição - As variaiveis que menos influenciaram são zeradas para não poluir o grafico
            df_contribuicao = circi.copy()
            for x in df_score_res["VARIAVEL"]:
                df_contribuicao.loc[:, x] = 0

            ######## Geração do grafico de contribuição ########
            df_contribuicao = df_contribuicao.join(
                pd.Series(list(eixo_x)).rename("timestamp"), how="right"
            )

            df_contribuicao = df_contribuicao.to_json(orient="columns")
            df_score_prin = df_score_prin.to_json(orient="columns")
            df_score_dec = df_score_dec.to_json(orient="columns")

            return (
                json.loads(df_score_prin),
                json.loads(df_contribuicao),
                json.loads(df_score_dec),
                df_projection_full
            )

        except ValueError as err:
            print(err)
            raise err