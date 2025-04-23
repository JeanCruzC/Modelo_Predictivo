# -*- coding: utf-8 -*-
"""Forecast Multimodelo con Streamlit"""

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

# --- Carga de datos -----------------------------------------------------
uploaded = st.file_uploader(
    "Sube tu archivo Excel o CSV con datos de serie temporal", type=["xlsx", "csv"]
)
if not uploaded:
    st.stop()
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# --- Configuración de la columna de fechas -------------------------------
col_fecha = st.text_input("Nombre de la columna de fechas", value="Fecha")
if col_fecha not in df.columns:
    st.error(
        f"La columna '{col_fecha}' no existe. Columnas disponibles: {list(df.columns)}"
    )
    st.stop()

df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
df = df.set_index(col_fecha).ffill()

# --- Definir horizontes --------------------------------------------------
pasos_forecast = st.number_input(
    "¿Cuántos pasos hacia adelante quieres pronosticar?", min_value=1, value=6
)

# --- Selección de modelos ------------------------------------------------
all_models = ["SARIMA", "ARIMA", "WES", "RandomForest", "XGBoost", "Stacking"]
select_all = st.checkbox("Incluir todos los modelos", value=True)
selected_models = (
    all_models if select_all else st.multiselect(
        "Selecciona modelos", all_models, default=all_models
    )
)
if not selected_models:
    st.error("Selecciona al menos un modelo.")
    st.stop()

# --- Funciones auxiliares ------------------------------------------------
def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    diffs = series.index.to_series().diff().dropna()
    days = diffs.dt.days.mean()
    return 'D' if days <= 1.5 else 'W' if days <= 8 else 'MS'

def calcular_estacionalidad(data):
    n = len(data)
    if n >= 24:
        return 12
    if n >= 18:
        return 6
    if n >= 12:
        return 4
    if n >= 8:
        return 2
    return 1

def aplicar_log(serie):
    if serie.std() / serie.mean() > 0.2:
        return np.log1p(serie), True
    return serie, False

def walk_forward(train, test, seasonal, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            model = auto_arima(
                history, seasonal=seasonal, m=m,
                stepwise=True, suppress_warnings=True
            )
            p = model.predict(n_periods=1)[0]
        except:
            p = history.iloc[-1]
        preds.append(p)
        new_point = pd.Series([test.iloc[t]], index=[test.index[t]])
        history = pd.concat([history, new_point])
    return pd.Series(preds, index=test.index)

def walk_forward_wes(train, test, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            model = ExponentialSmoothing(
                history, seasonal='add', trend='add', seasonal_periods=m
            ).fit()
            p = model.forecast(1)[0]
        except:
            p = history.iloc[-1]
        preds.append(p)
        new_point = pd.Series([test.iloc[t]], index=[test.index[t]])
        history = pd.concat([history, new_point])
    return pd.Series(preds, index=test.index)

def forecast_multimodelo(
    data, m, freq, steps, rf_model, xgb_model, ridge_model, selected_models
):
    fechas = pd.date_range(
        start=data.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=steps, freq=freq
    )
    seasonal_flag = (m > 1 and len(data) >= 2*m)
    sarima = auto_arima(
        data, seasonal=seasonal_flag,
        m=(m if seasonal_flag else 1),
        stepwise=True, suppress_warnings=True
    ).predict(n_periods=steps)
    arima = auto_arima(
        data, seasonal=False,
        stepwise=True, suppress_warnings=True
    ).predict(n_periods=steps)
    try:
        wes = ExponentialSmoothing(
            data, seasonal='add', trend='add', seasonal_periods=m
        ).fit().forecast(steps)
    except:
        wes = np.repeat(data.iloc[-1], steps)
    # Features ML
    hist = data.copy()
    feats = []
    for i in range(1, steps+1):
        fdate = data.index[-1] + pd.tseries.frequencies.to_offset(freq)*i
        feats.append([
            fdate.month,
            fdate.dayofyear,
            hist.iloc[-1],
            hist.iloc[-2] if len(hist)>1 else hist.iloc[-1],
            hist.iloc[-3:].mean() if len(hist)>=3 else hist.iloc[-1]
        ])
        hist = pd.concat([hist, pd.Series([hist.iloc[-1]], index=[fdate])])
    Xf = pd.DataFrame(
        feats,
        columns=['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3'],
        index=fechas
    )
    rf_pred = rf_model.predict(Xf) if 'RandomForest' in selected_models else None
    xgb_pred = xgb_model.predict(Xf) if 'XGBoost' in selected_models else None
    stack_input = pd.DataFrame({
        'SARIMA': sarima,
        'ARIMA':  arima,
        'WES':    wes,
        **({'RandomForest': rf_pred} if rf_pred is not None else {}),
        **({'XGBoost': xgb_pred} if xgb_pred is not None else {})
    }, index=fechas)
    stack = ridge_model.predict(stack_input) if 'Stacking' in selected_models else None
    df_res = pd.DataFrame({
        'SARIMA': sarima,
        'ARIMA':  arima,
        'WES':    wes,
        **({'RandomForest': rf_pred} if rf_pred is not None else {}),
        **({'XGBoost': xgb_pred} if xgb_pred is not None else {}),
        **({'Stacking': stack} if stack is not None else {})
    }, index=fechas)
    return df_res

# --- Preparación inicial ------------------------------------------------
freq = detectar_frecuencia(df.iloc[:,0])
m = calcular_estacionalidad(df.iloc[:,0])
series, log_flag = aplicar_log(df.iloc[:,0])
if log_flag:
    df.iloc[:,0] = series

# --- División Train/Test ------------------------------------------------
train = df.iloc[:int(len(df)*0.8), 0]
test  = df.iloc[int(len(df)*0.8):, 0]

# --- Tuning ML ----------------------------------------------------------
rf_model = None
xgb_model = None
ridge_model = Ridge()

if 'RandomForest' in selected_models:
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators':[100,200],'max_depth':[5,10,None]},
        cv=3
    )
    rf.fit(train.to_frame(name='Valor'), train)
    rf_model = rf
if 'XGBoost' in selected_models:
    xgb = GridSearchCV(
        XGBRegressor(random_state=42),
        {'n_estimators':[100,200],'learning_rate':[0.05,0.1]},
        cv=3
    )
    xgb.fit(train.to_frame(name='Valor'), train)
    xgb_model = xgb

# --- Walk-Forward Validation ------------------------------------------
wfv = pd.DataFrame(index=test.index)
if 'SARIMA' in selected_models:
    wfv['SARIMA'] = walk_forward(train, test, seasonal=(m>1), m=m)
if 'ARIMA' in selected_models:
    wfv['ARIMA']  = walk_forward(train, test, seasonal=False, m=m)
if 'WES' in selected_models:
    wfv['WES']    = walk_forward_wes(train, test, m)

st.subheader("Resultados Walk-Forward")
if not wfv.empty:
    st.line_chart(wfv)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        wfv.to_excel(writer, sheet_name='WFV')
    st.download_button(
        "Descargar Walk-Forward", buf.getvalue(),
        file_name="predicciones_walk_forward.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Pronóstico a futuro -----------------------------------------------
if ('RandomForest' in selected_models and rf_model is None) or \
   ('XGBoost' in selected_models and xgb_model is None):
    st.error("Falta tuning de ML para pronóstico futuro.")
else:
    fut = forecast_multimodelo(
        df.iloc[:,0], m, freq, pasos_forecast,
        rf_model, xgb_model, ridge_model, selected_models
    )
    if log_flag:
        fut = np.expm1(fut)
    st.subheader("Forecast Multimodelo")
    st.line_chart(fut)
    buf2 = BytesIO()
    with pd.ExcelWriter(buf2, engine='openpyxl') as writer:
        fut.to_excel(writer, sheet_name='Forecast')
    st.download_button(
        "Descargar Forecast", buf2.getvalue(),
        file_name="forecast_multimodelo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )