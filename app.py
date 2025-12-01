import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

st.set_page_config(
    page_title="Previsão do preço do petróleo Brent (FOB)",
    layout="wide"
)

st.title("📈 Previsão do preço do petróleo Brent (FOB)")
st.write(
    """
    Aplicação desenvolvida para a Pós-Tech Data Analytics – Fase 4 (Prova Substitutiva).

    A série histórica é obtida automaticamente do **Ipeadata**.
    O modelo ARIMA é treinado dinamicamente na aplicação.
    """
)

# ===============================
# 1. Carregar dados
# ===============================

@st.cache_data
def carregar_dados():
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
    tabelas = pd.read_html(url, decimal=",", thousands=".")

    df_raw = tabelas[2].copy()
    df_raw.columns = ["Data", "Preco_Brent_FOB"]
    df_raw = df_raw[df_raw["Data"] != "Data"]

    df_raw["Data"] = pd.to_datetime(df_raw["Data"], dayfirst=True)
    df_raw["Preco_Brent_FOB"] = df_raw["Preco_Brent_FOB"].astype(float)

    df = df_raw.sort_values("Data").dropna()
    df = df.drop_duplicates(subset="Data")
    df = df.set_index("Data")

    return df


df = carregar_dados()

st.success("Dados carregados com sucesso!")

# ===============================
# 2. Treinar modelo ARIMA
# ===============================

@st.cache_resource
def treinar_modelo(df):
    modelo = auto_arima(
        df["Preco_Brent_FOB"],
        seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True
    )
    return modelo

with st.spinner("Treinando modelo ARIMA..."):
    modelo = treinar_modelo(df)

st.success("Modelo treinado!")

# ===============================
# 3. Previsão futura
# ===============================

def prever(modelo, df, dias):
    previsoes = modelo.predict(n_periods=dias)
    datas = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=dias)
    return pd.DataFrame({"Data": datas, "Previsao": previsoes})

dias = st.slider("Dias futuros para prever:", 7, 180, 60)

previsao_df = prever(modelo, df, dias)

# ===============================
# 4. Gráficos
# ===============================

tab1, tab2 = st.tabs(["Histórico completo", "Zoom + Previsão"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df.index, df["Preco_Brent_FOB"], label="Histórico")
    ax1.plot(previsao_df["Data"], previsao_df["Previsao"], label=f"Previsão ({dias} dias)")
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    janela = 365
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df.iloc[-janela:].index, df.iloc[-janela:]["Preco_Brent_FOB"], label="Últimos 12 meses")
    ax2.plot(previsao_df["Data"], previsao_df["Previsao"], label=f"Previsão ({dias} dias)")
    ax2.legend()
    st.pyplot(fig2)

# ===============================
# 5. Tabela
# ===============================

st.subheader("📄 Tabela da previsão")
st.dataframe(previsao_df.set_index("Data"))

st.caption("Fonte: Ipeadata – série EIA366_PBRENT366")
