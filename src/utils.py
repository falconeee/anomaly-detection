import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

from .preprocessing import drop_transitorio_desligado

warnings.filterwarnings("ignore")

def graph_variables(df, eixo ,variaveis=[], freq="1min", pre_process=None, list_periods_anom=None, df_projection=None):
    ## Converte eixoX para lista
    df["timestamp"] = eixo.to_list()
    
    ## Criando lista de períodos desligados
    list_periods_off = []
    if pre_process:
        for pro in pre_process:
            _,_,list_aux = drop_transitorio_desligado(df,pro["variable_off"],pro["limit_off"],pro["interval_off"],"timestamp",pre_corte=pro["pre_cut"],pos_corte=pro["after_cut"])
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

def graph_predict(phi, eixo_x, threshold, start_date, end_date, freq="1min", list_periods=None):
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

    return fig
