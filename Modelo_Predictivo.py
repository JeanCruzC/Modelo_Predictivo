# -*- coding: utf-8 -*-
"""Forecast Multimodelo con Streamlit (con caché para mayor rendimiento)"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import BytesIO
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

st.title("Forecast Multimodelo con Streamlit")

# --- Funciones cacheadas ------------------------------------------------
@st.cache_data(show_spinner=False)
def load_df(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

@st.cache_data(show_spinner=False)
def preprocess(df, col_fecha):
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    return df.set_index(col_fecha).ffill()

@st.cache_data(show_spinner="Detectando frecuencia...")
def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    diffs = series.index.to_series().diff().dropna()
    days = diffs.dt.days.mean()
    return 'D' if days<=1.5 else 'W' if days<=8 else 'MS'

@st.cache_data(show_spinner="Calculando estacionalidad...")
def calcular_estacionalidad(series):
    n = len(series)
    if n>=24: return 12
    if n>=18: return 6
    if n>=12: return 4
    if n>=8:  return 2
    return 1

@st.cache_data(show_spinner="Aplicando log si es necesario...")
def aplicar_log(series):
    if series.std()/series.mean()>0.2:
        return np.log1p(series), True
    return series, False

@st.cache_data(show_spinner="Ajustando SARIMA...")
def fit_sarima(series, seasonal, m):
    return auto_arima(
        series,
        seasonal=seasonal,
        m=(m if seasonal else 1),
        stepwise=True,
        suppress_warnings=True
    )

@st.cache_data(show_spinner="Ajustando WES...")
def fit_wes(series, m):
    model = ExponentialSmoothing(
        series,
        seasonal='add',
        trend='add',
        seasonal_periods=m
    ).fit()
    return model

@st.cache_data(show_spinner="Buscando RandomForest...")
def tune_rf(X, y):
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators':[100,200],'max_depth':[5,10,None]},
        cv=3,
        n_jobs=-1
    )
    gs.fit(X, y)
    return gs

@st.cache_data(show_spinner="Buscando XGBoost...")
def tune_xgb(X, y):
    gs = GridSearchCV(
        XGBRegressor(random_state=42),
        {'n_estimators':[100,200],'learning_rate':[0.05,0.1]},
        cv=3,
        n_jobs=-1
    )
    gs.fit(X, y)
    return gs

# --- Funciones de pronóstico --------------------------------------------

def walk_forward(train, test, sarima_model, wes_model, m):
    history = train.copy()
    preds_sarima, preds_wes = [], []
    for t in range(len(test)):
        # SARIMA predict one step
        try:
            p1 = sarima_model.predict(n_periods=1)[0]
        except:
            p1 = history.iloc[-1]
        preds_sarima.append(p1)
        # WES forecast one step
        try:
            p2 = wes_model.forecast(1)[0]
        except:
            p2 = history.iloc[-1]
        preds_wes.append(p2)
        # update history
        new_point = pd.Series([test.iloc[t]], index=[test.index[t]])
        history = pd.concat([history, new_point])
    return pd.Series(preds_sarima, index=test.index), pd.Series(preds_wes, index=test.index)

# --- Interfaz -----------------------------------------------------------
uploaded = st.file_uploader(
    "Sube tu archivo (Excel o CSV)", type=["xlsx","csv"]
)
if not uploaded:
    st.info("Por favor, sube un archivo de datos.")
    st.stop()

raw = load_df(uploaded)
col_fecha = st.text_input("Columna de fecha", value=raw.columns[0])
if col_fecha not in raw.columns:
    st.error("Columna de fecha no encontrada.")
    st.stop()

df = preprocess(raw, col_fecha)
series = df.iloc[:,0]

freq = detectar_frecuencia(series)
m = calcular_estacionalidad(series)
series, log_flag = aplicar_log(series)

pasos = st.number_input("Pasos a pronosticar", min_value=1, value=6)

# División train/test
split = int(len(series)*0.8)
train, test = series[:split], series[split:]

# Ajuste de modelos pesados solo una vez
sarima_model = fit_sarima(train, seasonal=(m>1), m=m)
wes_model    = fit_wes(train, m)

# ML tuning bajo demanda
data_ml = pd.DataFrame({
    'Valor': series
}).assign(
    Mes=series.index.month,
    DiaDelAnio=series.index.dayofyear,
    Lag1=series.shift(1),
    Lag2=series.shift(2),
    MediaMovil3=series.rolling(3).mean()
).dropna()

if 'RandomForest' in st.multiselect("Modelos ML a ajustar", ['RandomForest','XGBoost']):
    X = data_ml[['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']]
    y = data_ml['Valor']
    rf_model = tune_rf(X, y)
    xgb_model = tune_xgb(X, y)
else:
    rf_model = xgb_model = None

# Walk-forward
st.subheader("Validación Walk-Forward")
pred_sarima, pred_wes = walk_forward(train, test, sarima_model, wes_model, m)
results = pd.DataFrame({
    'SARIMA': pred_sarima,
    'WES': pred_wes
}, index=test.index)
st.line_chart(results)

# Forecast futuro
st.subheader("Forecast a futuro")
tonf = sarima_model.predict(n_periods=pasos)
wesf = wes_model.forecast(pasos)

# Pred ML si existen
fdates = pd.date_range(start=series.index[-1]+pd.tseries.frequencies.to_offset(freq), periods=pasos, freq=freq)
ml_preds = {}
if rf_model:
    Xf = data_ml[['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']].iloc[-pasos:]
    ml_preds['RF'] = rf_model.predict(Xf)
if xgb_model:
    ml_preds['XGB'] = xgb_model.predict(Xf)

# Stacking
stack_df = pd.DataFrame({
    'SARIMA': tonf,
    'WES': wesf,
    **ml_preds
}, index=fdates)
ridge = Ridge().fit(stack_df.fillna(method='ffill'), np.zeros(len(stack_df)))
stackf = ridge.predict(stack_df.fillna(method='ffill'))
stack_df['Stack'] = stackf
st.line_chart(stack_df)

# Descargas
buf = BytesIO()
with pd.ExcelWriter(buf, engine='openpyxl') as writer:
    results.to_excel(writer, sheet_name='WFV')
    stack_df.to_excel(writer, sheet_name='Forecast')
st.download_button("Descargar resultados", buf.getvalue(), "forecast_multimodelo.xlsx")
