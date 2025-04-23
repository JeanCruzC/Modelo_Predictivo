# -*- coding: utf-8 -*-
"""Forecast Multimodelo con Streamlit (con caché, logging, y fallback robusto)"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import BytesIO
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# --- Logging en sidebar ---
log_container = st.sidebar.empty()
def log(msg):
    log_container.text(msg)

st.title("Forecast Multimodelo con Streamlit")
st.write("Observa la barra lateral para el progreso de cada etapa")

# --- Funciones cacheadas ---
@st.cache_data(show_spinner=False)
def load_df(uploaded_file):
    log("Cargando datos...")
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    log("Datos cargados")
    return df

@st.cache_data(show_spinner=False)
def preprocess(df, col_fecha):
    log(f"Preprocesando columna de fecha: {col_fecha}")
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df2 = df.set_index(col_fecha).ffill()
    log("Preprocesado completado")
    return df2

@st.cache_data(show_spinner=False)
def detectar_frecuencia(series):
    log("Detectando frecuencia de la serie...")
    freq = pd.infer_freq(series.index)
    if not freq:
        diffs = series.index.to_series().diff().dropna()
        days = diffs.dt.days.mean()
        freq = 'D' if days<=1.5 else 'W' if days<=8 else 'MS'
    log(f"Frecuencia detectada: {freq}")
    return freq

@st.cache_data(show_spinner=False)
def calcular_estacionalidad(series):
    log("Calculando estacionalidad...")
    n = len(series)
    m = 12 if n>=24 else 6 if n>=18 else 4 if n>=12 else 2 if n>=8 else 1
    log(f"Estacionalidad (m): {m}")
    return m

@st.cache_data(show_spinner=False)
def aplicar_log(series):
    log("Revisando necesidad de transformación log...")
    if series.std() / max(series.mean(), 1e-9) > 0.2:
        log("Aplicando transformación logarítmica")
        return np.log1p(series), True
    log("No se aplica logarítmico")
    return series, False

@st.cache_data(show_spinner=False)
def fit_sarima(series, seasonal, m):
    log("Ajustando modelo SARIMA...")
    model = auto_arima(
        series,
        seasonal=seasonal,
        m=(m if seasonal else 1),
        stepwise=True,
        suppress_warnings=True
    )
    log("SARIMA ajustado")
    return model

@st.cache_data(show_spinner=False)
def fit_wes(series, m):
    log("Ajustando modelo WES (Holt-Winters)...")
    try:
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=m
        ).fit()
        log("WES ajustado con estacionalidad")
        return model
    except Exception as e:
        log("Error WES estacional, aplicando fallback sin estacionalidad")
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal=None
        ).fit()
        log("WES ajustado sin estacionalidad")
        return model

@st.cache_data(show_spinner=False)
def tune_rf(X, y):
    log("Iniciando tuning RandomForest...")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators': [100,200], 'max_depth': [5,10,None]},
        cv=3,
        n_jobs=-1
    )
    gs.fit(X, y)
    log("RandomForest ajustado")
    return gs

@st.cache_data(show_spinner=False)
def tune_xgb(X, y):
    log("Iniciando tuning XGBoost...")
    gs = GridSearchCV(
        XGBRegressor(random_state=42),
        {'n_estimators': [100,200], 'learning_rate': [0.05,0.1]},
        cv=3,
        n_jobs=-1
    )
    gs.fit(X, y)
    log("XGBoost ajustado")
    return gs

# --- Sidebar UI ---
with st.sidebar:
    uploaded = st.file_uploader("Sube tu archivo (Excel o CSV)", type=["xlsx","csv"])
    col_fecha = st.text_input("Columna de fecha", value=None)
    pasos = st.number_input("Pasos a pronosticar", min_value=1, value=6)
    select_ml = st.multiselect("Modelos ML a ajustar", ["RandomForest","XGBoost"])

if not uploaded or not col_fecha:
    st.stop()

# --- Pipeline de ejecución ---
raw = load_df(uploaded)
if col_fecha not in raw.columns:
    st.error("Columna de fecha no encontrada.")
    st.stop()

df = preprocess(raw, col_fecha)
series = df.iloc[:, 0]
freq = detectar_frecuencia(series)
m = calcular_estacionalidad(series)
series, log_flag = aplicar_log(series)

# Split train/test
t = len(series)
split = int(0.8 * t)
train, test = series[:split], series[split:]

# Ajuste de modelos pesados
sarima_model = fit_sarima(train, seasonal=(m>1), m=m)
wes_model    = fit_wes(train, m)

# ML tuning opcional
data_ml = None
rf_model = xgb_model = None
if 'RandomForest' in select_ml or 'XGBoost' in select_ml:
    data_ml = pd.DataFrame({'Valor': series}).assign(
        Mes=series.index.month,
        DiaDelAnio=series.index.dayofyear,
        Lag1=series.shift(1),
        Lag2=series.shift(2),
        MediaMovil3=series.rolling(3).mean()
    ).dropna()
    if 'RandomForest' in select_ml:
        rf_model = tune_rf(data_ml[['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']], data_ml['Valor'])
    if 'XGBoost' in select_ml:
        xgb_model = tune_xgb(data_ml[['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']], data_ml['Valor'])

# Walk-Forward Validation
st.subheader("Validación Walk-Forward")
progress = st.progress(0)
predictions = {'SARIMA': [], 'WES': []}
history = train.copy()
for i in range(len(test)):
    p_s = sarima_model.predict(n_periods=1)[0]
    p_w = wes_model.forecast(1)[0]
    predictions['SARIMA'].append(p_s)
    predictions['WES'].append(p_w)
    history = pd.concat([history, pd.Series([test.iloc[i]], index=[test.index[i]])])
    progress.progress((i+1)/len(test))
results = pd.DataFrame(predictions, index=test.index)
st.line_chart(results)
log("Walk-forward completado")

# Forecast futuro
st.subheader("Forecast a futuro")
with st.spinner("Generando pronóstico futuro..."):
    forecast_s = sarima_model.predict(n_periods=pasos)
    forecast_w = wes_model.forecast(pasos)

fdates = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=pasos, freq=freq)
stack_df = pd.DataFrame({'SARIMA': forecast_s, 'WES': forecast_w}, index=fdates)
# Agregar ML si existen
if rf_model:
    Xf = pd.DataFrame({
        'Mes': fdates.month,
        'DiaDelAnio': fdates.dayofyear,
        'Lag1': series.iloc[-1],
        'Lag2': series.iloc[-2] if len(series)>1 else series.iloc[-1],
        'MediaMovil3': series.iloc[-3:].mean() if len(series)>=3 else series.iloc[-1]
    }, index=fdates)
    stack_df['RF'] = rf_model.predict(Xf)
if xgb_model:
    stack_df['XGB'] = xgb_model.predict(Xf)
# Stacking
ridge = Ridge().fit(stack_df.fillna(method='ffill'), np.zeros(len(stack_df)))
stack_df['Stack'] = ridge.predict(stack_df.fillna(method='ffill'))
st.line_chart(stack_df)
log("Forecast completado")

# Descarga de resultados
buf = BytesIO()
with pd.ExcelWriter(buf, engine='openpyxl') as writer:
    results.to_excel(writer, sheet_name='WalkForward')
    stack_df.to_excel(writer, sheet_name='Forecast')
st.download_button("Descargar resultados", buf.getvalue(), "forecast_multimodelo.xlsx")
