# Anomaly Detection

Detecção de anomalias não supervisionada em séries temporais multivariadas de
processos industriais (bombas, ventiladores, motores etc.), com análise de causa
raiz por variável.

O repositório reúne oito modelos de deep learning, todos com a mesma interface
`fit` / `predict` / `contribution`. O limiar de anomalia é calculado
automaticamente via **SPOT** (Teoria de Valores Extremos). Assim, os modelos
podem ser comparados lado a lado sobre os mesmos dados de processo.

## Modelos

| Modelo | Arquivo | Arquitetura | Referência |
|---|---|---|---|
| **MSCVAE** | `src/models/MSCVAE.py` | VAE convolucional sobre matrizes de correlação multiescala (janelas 10/30/60) + transformer espaço-temporal | Inspirado em MSCRED |
| **MSCRED** | `src/models/MSCRED.py` | Matrizes de assinatura + ConvLSTM com atenção temporal | Zhang et al., AAAI 2019 |
| **USAD** | `src/models/USAD.py` | Autoencoders adversariais com encoder compartilhado | Audibert et al., KDD 2020 |
| **OmniAnomaly** | `src/models/OmniAnomaly.py` | VAE estocástico recorrente com *planar normalizing flows* | Su et al., KDD 2019 |
| **TranAD** | `src/models/TranAD.py` | Transformer com treinamento adversarial em duas fases | Tuli et al., VLDB 2022 |
| **ANN_AE** | `src/models/ANN_AE.py` | Autoencoder denso (MLP) sobre janelas deslizantes | — |
| **CNN_AE** | `src/models/CNN_AE.py` | Autoencoder convolucional 1D | — |
| **LSTM_AE** | `src/models/LSTM_AE.py` | Autoencoder LSTM | — |

O limiar de todos os modelos vem do **SPOT** (Siffer et al.,
KDD 2017). Ele ajusta uma distribuição Generalizada de Pareto à cauda dos erros
de reconstrução no treino. O parâmetro `gain` multiplica esse limiar para
ajustar a sensibilidade manualmente.

## Estrutura

```
anomaly-detection/
├── src/
│   ├── models/            # Os 8 modelos de deep learning
│   ├── preprocessing.py   # Pré-processamento: carga, ON/OFF, transitórios, NaN, tags
│   └── utils.py           # Gráficos (Plotly)
├── examples/              # Notebooks de aplicação em ativos reais
│   ├── bomba_392P2007.ipynb
│   ├── bomba_392P2048-49.ipynb
│   └── ventilador_forno_RBR-W7.ipynb
├── legacy/                # Código antigo mantido apenas como referência
│   ├── models_v0/         # Primeira versão dos modelos
│   └── AE-tests.ipynb     # Notebook exploratório (usa APIs antigas, não roda com src/ atual)
├── data/                  # Dados de processo (fora do git, ver abaixo)
├── pyproject.toml
└── uv.lock
```

## Instalação

O projeto usa Python 3.12 e [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Isso cria o `.venv` com PyTorch, pandas, scikit-learn, SciPy, Plotly, Captum e
`ipykernel` (para rodar os notebooks). Em Apple Silicon os modelos usam `mps`
automaticamente; com GPU NVIDIA, `cuda`; caso contrário, CPU.

## Dados

A pasta `data/` **não é versionada** (está no `.gitignore`). Cada ativo precisa
de dois arquivos com o mesmo prefixo (`process_name`):

**`<process_name>.parquet`**: série temporal principal. Uma coluna de timestamp
(o nome é configurável no notebook, ex. `Timestamp` ou `TIMESTAMP`) e uma coluna
numérica por tag/sensor. CSV também é aceito (`sep=";"`).

**`<process_name>_subsistema.csv`**: descrição das tags, separada por `;`:

```csv
VARIAVEL;DESC;SISTEMA
392P2007.05HA.BM.LA;Aceleração Horizontal Lado Acionado Bomba;392P2007 Bomba
392P2007.05HV.BM.LA;Velocidade Horizontal Lado Acionado Bomba;392P2007 Bomba
```

`VARIAVEL` deve bater com o nome da coluna no parquet. `SISTEMA` agrupa as tags
por subsistema na análise de contribuição.

## Uso

O fluxo típico (é o que os notebooks em `examples/` fazem) é:

```python
import src.preprocessing as ppd
import src.utils as utils
from src.models.MSCVAE import MSCVAE

timestamp = "Timestamp"

# 1. Carga e descrição das tags
df = ppd.load_dataset_principal("data/392P2007 Bomba.parquet", ["392M3820_ST"], timestamp, dropna=True)
df_sistema, _ = ppd.set_tags_config(df, "data/392P2007 Bomba_subsistema.csv")

# 2. Períodos de treino (operação normal), sem os momentos de equipamento
#    desligado e sem os transitórios de partida/parada
df_train_list = []
for inicio, fim in [("2025-09-26", "2025-09-30"), ("2025-12-12 00:00", "2025-12-12 04:00")]:
    df_p = df[(df[timestamp] >= inicio) & (df[timestamp] <= fim)]
    df_p, _, _ = ppd.drop_transitorio_desligado(
        df_p, "392P2007_ST", 400, 0, timestamp, pre_corte=90, pos_corte=150
    )
    df_train_list.append(df_p.drop(columns=timestamp).reset_index(drop=True))

# 3. Treino: o limiar SPOT é calculado ao final do fit
model = MSCVAE()
model.fit(df_train_list, epochs=50, gain=1.0)

# 4. Inferência sobre o período de teste
df_test = df[(df[timestamp] > "2026-01-01") & (df[timestamp] <= "2026-05-01")]
df_test, _, _ = ppd.drop_transitorio_desligado(df_test, "392P2007_ST", 400, 0, timestamp, pre_corte=90, pos_corte=150)
eixo_x = df_test[timestamp]
df_test = df_test.drop(columns=timestamp)

pred = model.predict(df_test, timestamps=eixo_x)   # {"timestamp": [...], "phi": [...]}
anomalias = [t for t, phi in zip(pred["timestamp"], pred["phi"]) if phi > model.threshold]

# 5. Causa raiz: contribuição de cada variável + série reconstruída
contribution, df_reconstruction = model.contribution(df_test, df_sistema)
```

### Interface comum

| Método | Retorno |
|---|---|
| `fit(df_train, epochs, gain, ...)` | Treina o modelo e define `model.threshold`. Aceita um `DataFrame` ou uma lista de `DataFrame`s (períodos de treino não contíguos). |
| `predict(df_test, timestamps=None)` | `dict` com `timestamp` e `phi` (score de anomalia). `phi > threshold` indica anomalia. |
| `contribution(df_test, df_sistema, ...)` | `(contribution, df_reconstruction)`: contribuição por variável/subsistema e o `DataFrame` reconstruído na escala original. |

Os hiperparâmetros de arquitetura (`seq_len`, `latent_dim`, `window_sizes`
etc.) ficam no construtor de cada classe, e todos aceitam `seed` (padrão 42)
para reprodutibilidade.

### Pré-processamento (`src/preprocessing.py`)

| Função | Para quê |
|---|---|
| `load_dataset_principal` | Lê parquet/CSV, remove colunas, converte timestamp e tipos, ordena |
| `set_tags_config` | Cruza o arquivo de subsistema com as colunas e calcula IQR/Q3 por tag |
| `drop_transitorio_desligado` | Remove períodos com o equipamento desligado (variável abaixo de um limite) e os transitórios antes/depois |
| `select_periods_nan` | Lista os períodos sem dados, para sombrear nos gráficos |
| `merge_periods` | Une intervalos sobrepostos |

### Visualização (`src/utils.py`)

- `graph_predict`: série de `phi` contra o limiar, com períodos de anomalia destacados.
- `graph_variables`: séries originais × reconstruídas por variável.

## Notebooks

Cada notebook em `examples/` aplica todos os modelos a um ativo e compara os
resultados. Eles esperam ser executados de dentro de `examples/` (adicionam a
raiz do repositório ao `sys.path`) e leem os dados de `../data/`.

O projeto inclui apenas o `ipykernel`: abra os notebooks no VS Code (ou em um
Jupyter já instalado) e selecione o kernel do `.venv`.
