import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
from scipy.stats import invgauss
from scipy.stats.distributions import chi2
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def select_training_period(df_dataset, timestamp):
    df_np = df_dataset.copy()
    time = df_np[timestamp].to_list()
    df_np = df_np.drop(timestamp, axis=1).copy()
    array_np = df_np.to_numpy()

    ##PCA
    scaler = StandardScaler()
    # Fit on training set only.
    array_np_std = scaler.fit_transform(array_np)
    cov = np.cov(array_np_std.T)
    u, s, vh = np.linalg.svd(cov)
    pca = PCA(0.95)
    pca.fit(array_np_std)
    pc = pca.transform(array_np_std)
    nc = pc.shape[1]
    s_diag = np.diag(s)
    s_pcs = s_diag[:nc, :nc]

    ##T2
    t2 = []
    for i in range(pc.shape[0]):
        termo1 = pc[i]
        termo2 = np.linalg.inv(s_pcs)
        termo3 = pc[i].T

        t2.append(termo1.dot(termo2).dot(termo3))
    M = pc.shape[1]
    N = pc.shape[0]
    F = scipy.stats.f.ppf(0.95, M, N - M)
    t2_lim = (M * (N - 1) / (N - M)) * F

    ##SPE
    spe = []
    for i in range(pc.shape[0]):
        rs = array_np_std[i].dot(u[:, nc - 1 :])
        termo1 = rs.T
        termo2 = rs
        spe.append(termo1.dot(termo2))
    teta1 = (s_diag[nc - 1 :]).sum()
    teta2 = (s_diag[nc - 1 :] ** 2).sum()
    teta3 = (s_diag[nc:-1, :] ** 3).sum()
    h0 = 1 - (2 * teta1 * teta3) / (3 * teta2**2)
    mu = 0.145462645553
    vals = invgauss.ppf([0, 0.999], mu)
    ca = invgauss.cdf(vals, mu)[1]
    spe_lim = teta1 * (
        (h0 * ca * np.sqrt(2 * teta2) / teta1)
        + 1
        + (teta2 * h0 * (h0 - 1)) / (teta1**2)
    ) ** (1 / h0)

    ##PHI
    phi = []
    for i in range(pc.shape[0]):
        phi.append((spe[i] / spe_lim) + (t2[i] / t2_lim))
    gphi = ((nc / t2_lim**2) + (teta2 / spe_lim**2)) / (
        (nc / t2_lim) + (teta1 / spe_lim)
    )
    hphi = ((nc / t2_lim) + (teta1 / spe_lim)) ** 2 / (
        (nc / t2_lim**2) + (teta2 / spe_lim**2)
    )
    chi2.ppf(0.975, df=2)
    phi_lim = gphi * chi2.ppf(0.99, hphi)
    df_t2 = pd.DataFrame(
        {
            "time": time,
            "t2": t2,
            "spe": spe,
            "phi": phi,
        }
    )

    df_t2["t2_lim"] = t2_lim
    df_t2["spe_lim"] = spe_lim
    df_t2["phi_lim"] = phi_lim

    df_t2["t2"] = df_t2["t2"].ewm(alpha=0.01).mean()
    df_t2["spe"] = df_t2["spe"].ewm(alpha=0.01).mean()
    df_t2["phi"] = df_t2["phi"].ewm(alpha=0.01).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi"], mode="lines"))
    fig.add_trace(go.Scatter(x=df_t2["time"], y=df_t2["phi_lim"], mode="lines"))

    return fig

def graph_variables(df, eixo ,variaveis=[], freq="1T", pre_process=None, list_periods_anom=None, df_projection=None):
    ## Converte eixoX para lista
    df["timestamp"] = eixo.to_list()
    
    ## Criando lista de períodos desligados
    list_periods_off = []
    if pre_process:
        for pro in pre_process:
            _,_,list_aux = ppd.drop_transitorio_desligado(df,pro["variable_off"],pro["limit_off"],pro["interval_off"],"timestamp",pre_corte=pro["pre_cut"],pos_corte=pro["after_cut"])
            after_cut = pro["after_cut"]
            list_periods_off = [*list_periods_off,*list_aux]    
    
    ## Ajustando eixoX com amostras faltantes
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S")
    df = df.set_index("timestamp")
    df = df.resample(freq).asfreq()
    df = df.fillna(0)
    df = df.reset_index()
    eixo = df["timestamp"]
    df = df.drop("timestamp", axis=1)
    
    ## Cria objeto figura
    fig = go.Figure()
    
    ## Lista com 1 item = Univariado || Lista com de um item = Normalizado || Lista vazia = Normalizado com todas as variáveis
    if len(variaveis) == 1:
        ## Cria eixo Y com valor das variáveis
        y = [item for sublist in df[variaveis].values for item in sublist]
        fig.add_trace(go.Scatter(x=eixo, y=y, mode="lines", name=variaveis[0],line_color="#0F293A"))
        
        ## Definindo máximo e mínimo do eixo y
        y_max = max(y)
        y_min = min(y)
        ## df_projection = None então gráfico gerado tem projeção
        if df_projection is not None:
            ## Definindo novamente y_max caso necessário
            if max(df_projection[variaveis[0]]) > y_max:
                y_max = max(df_projection[variaveis[0]])
            ## Definindo novamente y_min caso necessário    
            if min(df_projection[variaveis[0]]) < y_min:
                y_min = min(df_projection[variaveis[0]])
            ## Verifica se anomalia se inicia com um desligado e corrige o número de amostras    
            if len(list_periods_off) > 0:
                if list_periods_off[0]["date_ini"] == eixo[0].to_pydatetime():
                    diff = list_periods_off[0]['date_end'] - list_periods_off[0]['date_ini']
                    diff_minutes = diff.total_seconds() / 60
                    df_projection = pd.concat([pd.DataFrame(np.nan, index=range(int(diff_minutes)+int(after_cut)), columns=df_projection.columns), df_projection], ignore_index=True)
                
            fig.add_trace(go.Scatter(x=eixo, y=df_projection[variaveis[0]], mode="lines", name="Projeção",line_color="red",line = dict(width=3, dash='dot')))
        
        
        ##### Adicionando sombreado nos períodos de ultrapassagem do limiar #####
        if list_periods_anom and list_periods_anom != []:
            ### Adiciona ponto no gráfico para aparecer na legenda ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_anom[0]["date_ini"]),
                y0=min(y),
                x1=str(list_periods_anom[0]["date_ini"]),
                y1=min(y),
                fillcolor='red',
                opacity=0.2,
                line_width=0,
                layer="below",
                name="Em Anomalia"

            )
            for periodo in list_periods_anom:
                fig.add_shape(
                    type="rect",
                    x0=str(periodo["date_ini"]),
                    y0=y_min,
                    x1=str(periodo["date_end"]),
                    y1=y_max,
                    fillcolor='red',
                    opacity=0.2,
                    line_width=0,
                    layer="below"
                )

        ##### Adicionando sombreado nos períodos de desligado #####
        if pre_process and list_periods_off != []:
            ### Adiciona ponto no gráfico para aparecer na legenda ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_off[0]["date_ini"]),
                y0=min(y),
                x1=str(list_periods_off[0]["date_ini"]),
                y1=min(y),
                fillcolor='#68cbf8',
                opacity=1,
                line_width=0,
                layer="below",
                name="Desligado"
            )
            fig.add_shape(
                    showlegend=True,
                    type="rect",
                    x0=str(list_periods_off[0]["date_ini"]),
                    y0=0,
                    x1=str(list_periods_off[0]["date_ini"]),
                    y1=0,
                    fillcolor='#b3e5fc',
                    opacity=1,
                    line_width=0,
                    layer="below",
                    name="Transitório",
                )
            
            for periodo in list_periods_off:
                if periodo["type"] == "desligado":
                    fig.add_shape(
                        type="rect",
                        x0=str(periodo["date_ini"]),
                        y0=y_min,
                        x1=str(periodo["date_end"]),
                        y1=y_max,
                        fillcolor='#68cbf8',
                        opacity=1,
                        line_width=0,
                        layer="below"
                    )
                if periodo["type"] == "transitorio":
                    fig.add_shape(
                        type="rect",
                        x0=str(periodo["date_ini"]),
                        y0=y_min,
                        x1=str(periodo["date_end"]),
                        y1=y_max,
                        fillcolor='#b3e5fc',
                        opacity=1,
                        line_width=0,
                        layer="below"
                    )     
        
        fig.update_layout(paper_bgcolor='white')
        fig.update_layout(plot_bgcolor='white')
        fig.update_layout(yaxis_range=[y_min,y_max])
        fig.update_layout(legend=dict(orientation="h"))
        fig.update_layout(showlegend=True)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        
    else:
        if len(variaveis) != 0:
            df = df.loc[:, variaveis]

        tamanho = df.shape
        start = 0
        end = 1
        x1 = eixo
        colunas = df.columns
        
        ## Adicionando sombreado nos períodos de desligado
        if pre_process and list_periods_off != []:
            ### Add a single dummy shape plot for the legend ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_off[0]["date_ini"]),
                y0=0,
                x1=str(list_periods_off[0]["date_ini"]),
                y1=0,
                fillcolor='#68cbf8',
                opacity=1,
                line_width=0,
                layer="below",
                name="Desligado"

            )
            for periodo in list_periods_off:
                if periodo["type"] == "desligado":
                    fig.add_shape(
                        type="rect",
                        x0=str(periodo["date_ini"]),
                        y0=0,
                        x1=str(periodo["date_end"]),
                        y1=tamanho[1],
                        fillcolor='#68cbf8',
                        opacity=1,
                        line_width=0,
                        layer="below"
                    )
                    
        ##### Adicionando sombreado nos períodos de ultrapassagem do limiar #####
        if list_periods_anom and list_periods_anom != []:
            ### Add a single dummy shape plot for the legend ###
            fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods_anom[0]["date_ini"]),
                y0=0,
                x1=str(list_periods_anom[0]["date_ini"]),
                y1=0,
                fillcolor='red',
                opacity=0.2,
                line_width=0,
                layer="below",
                name="Em Anomalia"

            )
            for periodo in list_periods_anom:
                fig.add_shape(
                    type="rect",
                    x0=str(periodo["date_ini"]),
                    y0=0,
                    x1=str(periodo["date_end"]),
                    y1=tamanho[1],
                    fillcolor='red',
                    opacity=0.2,
                    line_width=0,
                    layer="below"
                )

        for x in range(tamanho[1]):
            arr = df.iloc[:, x]
            width = end - start
            res = (arr - arr.min()) / np.ptp(arr) * width + start

            fig.add_trace(go.Scatter(x=x1, y=res, mode="lines", name=colunas[x]))
            fig.update_traces(line=dict(width=1))
            start = start + 1
            end = end + 1
        
        fig.update_layout(paper_bgcolor='white')
        fig.update_layout(plot_bgcolor='white')
        fig.update_layout(legend_traceorder="reversed")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CECFD1')
        
    return fig

def graph_predict(phi, eixo_x, threshold, process_id, start_date, end_date, freq="1T", list_periods=None, plot_anomalies=True):
    """
    Função para gerar o gráfico de detecção de anomalias e retornar o DataFrame `df_t2` com os dados de detecção.
    """

    # Criar o dataframe com os dados de phi e timestamps
    df = pd.DataFrame({"phi": phi, "timestamp": eixo_x})

    # Converter os timestamps para datetime e configurar o índice
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp")
    df = df.resample(freq).asfreq()
    df = df["phi"].fillna(0)
    df = df.reset_index()

    df["threshold"] = threshold

    # Criar o DataFrame df_t2 com as colunas 'time' e 'phi'
    df_t2 = pd.DataFrame({
        "time": df["timestamp"],  # Usando a coluna de timestamps
        "phi": df["phi"]  # Usando os valores de phi da predição
    })

    # Criar o gráfico
    layout = go.Layout(plot_bgcolor="white")
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["phi"], mode="lines", name="Índice", fill="tozeroy", line_color="#0F293A"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["threshold"], mode="lines", name="Limiar", line_color="#FB8102"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#CECFD1')
    fig.update_layout(hovermode='x unified', legend=dict(orientation="h"))
    fig.update_layout(yaxis_range=[0, 4 * threshold], paper_bgcolor='white', plot_bgcolor='white')

    # Ajustar end_date se for maior que o último timestamp
    last_timestamp = df["timestamp"].max()
    if end_date > last_timestamp:
        end_date = last_timestamp

    # Adicionar faixas de anomalias, se habilitado
    if plot_anomalies:
        anomalias_filt = get_anomalies(process_id, start_date, end_date)

        # Adicionar legendas fictícias
        fig.add_shape(type="rect", x0=(list_periods[0]["date_ini"]), x1=(list_periods[0]["date_ini"]), y0=0, y1=0, fillcolor="green", opacity=0.2,
                      line_width=0, layer="below", showlegend=True, name="Anomalia Real")
        fig.add_shape(type="rect", x0=(list_periods[0]["date_ini"]), x1=(list_periods[0]["date_ini"]), y0=0, y1=0, fillcolor="red", opacity=0.2,
                      line_width=0, layer="below", showlegend=True, name="Anomalia Falsa")

        # Plotar as anomalias
        for _, row in anomalias_filt.iterrows():
            color = 'green' if row["real_anomaly"] == 'true' else 'red'
            fig.add_shape(
                type="rect",
                x0=row["start_date"],
                x1=row["end_date"],
                y0=0,
                y1=df["phi"].max() * 4,
                fillcolor=color,
                opacity=0.2,
                line_width=0,
                layer="below"
            )

    # Adicionar faixas de períodos desligados/transitórios
    if list_periods and list_periods != []:
        fig.add_shape(showlegend=True, type="rect", x0=list_periods[0]["date_ini"], y0=0,
                      x1=list_periods[0]["date_ini"], y1=0, fillcolor='#68cbf8',
                      opacity=1, line_width=0, layer="below", name="Desligado")
        fig.add_shape(showlegend=True, type="rect", x0=list_periods[0]["date_ini"], y0=0,
                      x1=list_periods[0]["date_ini"], y1=0, fillcolor='#b3e5fc',
                      opacity=1, line_width=0, layer="below", name="Transitório")
        for periodo in list_periods:
            if periodo["type"] == "desligado":
                color = '#68cbf8'
            elif periodo["type"] == "transitorio":
                color = '#b3e5fc'
            else:
                color = "red"

            fig.add_shape(
                type="rect",
                x0=periodo["date_ini"],
                y0=0,
                x1=periodo["date_end"],
                y1=df["phi"].max() * 4,
                fillcolor=color,
                opacity=1,
                line_width=0,
                layer="below"
            )

    fig.update_xaxes(range=[start_date, end_date])

    return fig, df_t2
