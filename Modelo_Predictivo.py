# -*- coding: utf-8 -*-
"""Forecast Multimodelo con Streamlit y cache en AutoARIMA"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import BytesIO
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Supresión de warnings irrelevantes
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.title("Forecast Multimodelo con Streamlit")

# ------------------------------------------------------------
# Cache para auto_arima: evita recalcular el mismo modelo
# ------------------------------------------------------------
@st.cache_resource
def cached_auto_arima(series, seasonal, m):
    return auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True
    )

# === FUNCIONES AUXILIARES ===
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
            model = cached_auto_arima(history, seasonal, m)
            arr = model.predict(n_periods=1)
            p = arr.item() if hasattr(arr, 'item') else arr[0]
        except:
            p = history.iloc[-1]
        preds.append(p)
        next_pt = pd.Series([test.iloc[t]], index=[test.index[t]])
        history = pd.concat([history, next_pt])
    return pd.Series(preds, index=test.index)

def walk_forward_wes(train, test, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            model = ExponentialSmoothing(
                history,
                seasonal='add',
                trend='add',
                seasonal_periods=m
            ).fit()
            arr = model.forecast(1)
            p = arr.item() if hasattr(arr, 'item') else arr[0]
        except:
            p = history.iloc[-1]
        preds.append(p)
        next_pt = pd.Series([test.iloc[t]], index=[test.index[t]])
        history = pd.concat([history, next_pt])
    return pd.Series(preds, index=test.index)

def forecast_multimodelo(data, m, freq, steps, rf_model, xgb_model, ridge_model):
    fechas = pd.date_range(
        start=data.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=steps,
        freq=freq
    )
    seasonal_flag = (m > 1 and len(data) >= 2*m)

    # SARIMA
    sarima_model = cached_auto_arima(data, seasonal_flag, m)
    sarima_fc = sarima_model.predict(n_periods=steps)

    # ARIMA no estacional
    arima_model = cached_auto_arima(data, False, 0)
    arima_fc = arima_model.predict(n_periods=steps)

    # WES
    try:
        wes_fc = ExponentialSmoothing(
            data,
            seasonal='add',
            trend='add',
            seasonal_periods=m
        ).fit().forecast(steps)
    except:
        wes_fc = np.repeat(data.iloc[-1], steps)

    # Features para ML
    hist = data.copy()
    feats = []
    for i in range(1, steps + 1):
        dt = data.index[-1] + pd.tseries.frequencies.to_offset(freq) * i
        feats.append([
            dt.month,
            dt.dayofyear,
            hist.iloc[-1],
            hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1],
            hist.iloc[-3:].mean() if len(hist) >= 3 else hist.iloc[-1]
        ])
        hist = pd.concat([hist, pd.Series([hist.iloc[-1]], index=[dt])])
    Xf = pd.DataFrame(
        feats,
        columns=['Mes', 'DiaDelAnio', 'Lag1', 'Lag2', 'MediaMovil3'],
        index=fechas
    )

    rf_fc = rf_model.predict(Xf)
    xgb_fc = xgb_model.predict(Xf)

    df_stack = pd.DataFrame({
        'SARIMA': sarima_fc,
        'ARIMA': arima_fc,
        'WES': wes_fc,
        'RF': rf_fc,
        'XGB': xgb_fc
    }, index=fechas)
    stk_fc = ridge_model.predict(df_stack)

    df_res = pd.DataFrame({
        'SARIMA': sarima_fc,
        'ARIMA': arima_fc,
        'WES': wes_fc,
        'RandomForest': rf_fc,
        'XGBoost': xgb_fc,
        'Stacking': stk_fc
    }, index=fechas)
    return df_res

# === UI y lógica principal ===
# Carga de archivo
df_file = st.file_uploader(
    "Sube Excel/CSV con serie temporal",
    type=["xlsx", "csv"]
)
if not df_file:
    st.stop()
if df_file.name.endswith(".csv"):
    df0 = pd.read_csv(df_file)
else:
    df0 = pd.read_excel(df_file)

col_fecha = st.text_input("Columna de fechas", "Fecha")
if col_fecha not in df0.columns:
    st.error(f"Columna '{col_fecha}' no existe.")
    st.stop()

df0[col_fecha] = pd.to_datetime(df0[col_fecha], errors='coerce')
df0.set_index(col_fecha, inplace=True)
df0 = df0.ffill()

steps = st.number_input("Pasos de forecast", min_value=1, value=6)
models = ["SARIMA", "ARIMA", "WES", "RandomForest", "XGBoost", "Stacking"]
sel = st.multiselect("Modelos a usar", models, default=models)

series = df0.iloc[:, 0]
freq = detectar_frecuencia(series)
m = calcular_estacionalidad(series)
series_adj, logf = aplicar_log(series)
if logf:
    series = series_adj

data_train = series.iloc[:int(len(series) * 0.8)]
data_test = series.iloc[int(len(series) * 0.8):]

grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
    cv=2, n_jobs=-1
)
grid_rf.fit(data_train.to_frame('Valor'), data_train)
rf_m = grid_rf

grid_xgb = GridSearchCV(
XGBRegressor(random_state=42),
{'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
cv=2, n_jobs=-1
)
grid_xgb.fit(data_train.to_frame('Valor'), data_train)
xgb_m = grid_xgb

ridge_m = Ridge()

# Walk-Forward
wf = pd.DataFrame(index=data_test.index)
if 'SARIMA' in sel:
    wf['SARIMA'] = walk_forward(
data_train.to_frame('Valor'),
data_test.to_frame('Valor'),
seasonal=(m>1), m=m
)
if 'ARIMA' in sel:
    wf['ARIMA'] = walk_forward(
         data_train.to_frame('Valor'),
         data_test.to_frame('Valor'),
         seasonal=False, m=0
     )
if 'WES' in sel:
     wf['WES'] = walk_forward_wes(
         data_train.to_frame('Valor'),
         data_test.to_frame('Valor'),
         m
     )

st.subheader("Validación Walk-Forward")
st.line_chart(wf)
buf1 = BytesIO()
with pd.ExcelWriter(buf1, engine='openpyxl') as w:
     wf.to_excel(w, sheet_name='WFV')
st.download_button("Descargar WFV", buf1.getvalue(), file_name='wfv.xlsx')

# Forecast futuro
fc = forecast_multimodelo(series, m, freq, steps, rf_m, xgb_m, ridge_m)
if logf:
     fc = np.expm1(fc)

st.subheader("Forecast futuro")
st.line_chart(fc)
buf2 = BytesIO()
with pd.ExcelWriter(buf2, engine='openpyxl') as w:
     fc.to_excel(w, sheet_name='Forecast')
st.download_button("Descargar Forecast", buf2.getvalue(), file_name='forecast.xlsx')
