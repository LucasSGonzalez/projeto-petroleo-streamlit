import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Previsão do preço do petróleo Brent (FOB)",
    layout="wide"
)

st.title("Previsão do preço do petróleo Brent (FOB)")
st.write(
    """
    Aplicação desenvolvida para a Pós-Tech Data Analytics – Fase 4 (Prova Substitutiva).

    A série histórica é obtida automaticamente do **Ipeadata** e o modelo ARIMA foi 
    treinado previamente em notebook e carregado aqui via arquivo `.pkl`.
    """
)

# ===============================
# 1. Funções auxiliares
# ===============================

@st.cache_data
def carregar_dados():
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
    tabelas = pd.read_html(url, decimal=",", thousands=".")

    # Vamos chmar a tabela correta que é a de índice 2
    df_raw = tabelas[2].copy()
    df_raw.columns = ["Data", "Preco_Brent_FOB"]

    # Vamos remover linha de cabeçalho duplicado ("Data" como valor)
    df_raw = df_raw[df_raw["Data"] != "Data"]

    # Conversões
    df_raw["Data"] = pd.to_datetime(df_raw["Data"], dayfirst=True)
    df_raw["Preco_Brent_FOB"] = df_raw["Preco_Brent_FOB"].astype(float)

    # Ordenação e limpeza
    df = df_raw.sort_values("Data").dropna()
    df = df.drop_duplicates(subset="Data")
    df = df.set_index("Data")

    return df


@st.cache_resource
def carregar_modelo():
    modelo = joblib.load("modelo_petroleo_brent_arima.pkl")
    return modelo


def prever_preco_petroleo(modelo, serie, dias_futuros=30):
    previsoes = modelo.predict(n_periods=dias_futuros)

    datas_futuras = pd.date_range(
        start=serie.index[-1] + pd.Timedelta(days=1),
        periods=dias_futuros,
        freq="D"
    )

    return pd.DataFrame({
        "Data": datas_futuras,
        "Preco_Previsto": previsoes
    })


# ===============================
# 2. Carregar dados e modelo
# ===============================

with st.spinner("Carregando dados do Ipeadata..."):
    df = carregar_dados()

with st.spinner("Carregando modelo preditivo..."):
    modelo = carregar_modelo()

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.metric("Data inicial da série", str(df.index.min().date()))
with col_info2:
    st.metric("Data final da série", str(df.index.max().date()))

st.markdown("---")

# ===============================
# 3. Entrada do usuário
# ===============================

st.subheader("Configuração da previsão")

dias = st.slider(
    "Selecione o horizonte de previsão (em dias):",
    min_value=7,
    max_value=180,
    value=60,
    step=7
)

previsao = prever_preco_petroleo(modelo, df, dias_futuros=dias)

st.markdown("---")

# ===============================
# 4. Gráficos
# ===============================

st.subheader("Visualização da série histórica + previsão")

tab1, tab2 = st.tabs(["Histórico completo", "Zoom (últimos 12 meses)"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, df["Preco_Brent_FOB"], label="Histórico")
    ax1.plot(previsao["Data"], previsao["Preco_Previsto"],
             label=f"Previsão ({dias} dias)")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Preço (US$)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

with tab2:
    janela = 365
    data_inicio_zoom = df.index[-janela]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df.loc[data_inicio_zoom:].index,
             df.loc[data_inicio_zoom:, "Preco_Brent_FOB"],
             label="Histórico (últimos 12 meses)")
    ax2.plot(previsao["Data"], previsao["Preco_Previsto"],
             label=f"Previsão ({dias} dias)")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Preço (US$)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# ===============================
# 5. Tabela de previsões
# ===============================

st.subheader("Tabela detalhada das previsões")

st.dataframe(
    previsao.set_index("Data").round(2),
    use_container_width=True
)

st.caption(
    "Fonte dos dados históricos: Ipeadata – série EIA366_PBRENT366 (Preço por barril do petróleo bruto Brent, FOB, em US$)."
)
