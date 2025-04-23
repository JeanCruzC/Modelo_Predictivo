# -*- coding: utf-8 -*-
"""Forecast Multimodelo Streamlit – solo carga Excel, ejecuta en silencio y da descargas CSV"""

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
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Funciones auxiliares (idénticas a las del script headless)
# ---------------------------------------------------------------------
_arima_cache = {}

def cached_auto_arima(series, seasonal, m):
    key = (tuple(series.values), seasonal, m)
    if key not in _arima_cache:
        _arima_cache[key] = auto_arima(
            series,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
        )
    return _arima_cache[key]


def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    diffs = series.index.to_series().diff().dropna()
    days = diffs.dt.days.mean()
    if days <= 1.5:
        return "D"
    if days <= 8:
        return "W"
    return "MS"


def calcular_estacionalidad(series):
    n = len(series)
    if n >= 24:
        return 12
    if n >= 18:
        return 6
    if n >= 12:
        return 4
    if n >= 8:
        return 2
    return 1


def walk_forward(train, test, seasonal, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            p = float(cached_auto_arima(history, seasonal, m).predict(1)[0])
        except Exception:
            p = float(history.iloc[-1])
        preds.append(p)
        history = pd.concat([history, pd.Series(p, index=[test.index[t]])])
    return pd.Series(preds, index=test.index)


def walk_forward_wes(train, test, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            p = float(
                ExponentialSmoothing(
                    history,
                    seasonal="add",
                    trend="add",
                    seasonal_periods=m,
                )
                .fit()
                .forecast(1)[0]
            )
        except Exception:
            p = float(history.iloc[-1])
        preds.append(p)
        history = pd.concat([history, pd.Series(p, index=[test.index[t]])])
    return pd.Series(preds, index=test.index)


def forecast_multimodelo(series, m, freq, steps, rf_model, xgb_model, ridge_model):
    fechas = pd.date_range(
        start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=steps,
        freq=freq,
    )
    seasonal_flag = m > 1 and len(series) >= 2 * m
    sarima_fc = cached_auto_arima(series, seasonal_flag, m).predict(steps).astype(float)
    arima_fc = cached_auto_arima(series, False, 0).predict(steps).astype(float)
    try:
        wes_fc = (
            ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=m)
            .fit()
            .forecast(steps)
            .astype(float)
        )
    except Exception:
        wes_fc = np.repeat(float(series.iloc[-1]), steps)

    # ML features
    hist = series.copy()
    feats = []
    for dt in fechas:
        last = float(hist.iloc[-1])
        lag2 = float(hist.iloc[-2] if len(hist) > 1 else last)
        mv3 = float(hist.iloc[-3:].mean() if len(hist) >= 3 else last)
        feats.append([dt.month, dt.dayofyear, last, lag2, mv3])
        hist = pd.concat([hist, pd.Series(last, index=[dt])])

    Xf = pd.DataFrame(
        feats,
        index=fechas,
        columns=["Mes", "DiaDelAnio", "Lag1", "Lag2", "MediaMovil3"],
    )
    rf_fc = rf_model.predict(Xf).astype(float)
    xgb_fc = xgb_model.predict(Xf).astype(float)

    df_stack = pd.DataFrame(
        {
            "SARIMA": sarima_fc,
            "ARIMA": arima_fc,
            "WES": wes_fc,
            "RF": rf_fc,
            "XGB": xgb_fc,
        },
        index=fechas,
    )
    stk_fc = ridge_model.predict(df_stack).astype(float)

    return pd.DataFrame(
        {
            "SARIMA": sarima_fc,
            "ARIMA": arima_fc,
            "WES": wes_fc,
            "RandomForest": rf_fc,
            "XGBoost": xgb_fc,
            "Stacking": stk_fc,
        },
        index=fechas,
    )

# ---------------------------------------------------------------------
# UI mínima: solo carga del Excel y descarga de resultados
# ---------------------------------------------------------------------

st.title("Forecast Multimodelo – Procesamiento en Silencio")

uploaded = st.file_uploader("Sube tu archivo Excel (columna Fecha + columna Valor)", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.stop()

# Leer datos
df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") or uploaded.name.endswith(".xls") else pd.read_csv(uploaded)

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
df.set_index(df.columns[0], inplace=True)
series = df.iloc[:, 0].ffill()

# --- Pipeline silencioso ---
freq = detectar_frecuencia(series)
m = calcular_estacionalidad(series)

split = int(len(series) * 0.8)
train, test = series[:split], series[split:]

rf = GridSearchCV(RandomForestRegressor(random_state=42), {
    "n_estimators": [100], "max_depth": [5, None]
}, cv=3).fit(train.to_frame("Valor"), train)

xgb = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), {
    "n_estimators": [100], "learning_rate": [0.05]
}, cv=3).fit(train.to_frame("Valor"), train)

ridge = Ridge()

wf = pd.DataFrame(index=test.index)
wf["ARIMA"] = walk_forward(train, test, seasonal=False, m=0)
wf["SARIMA"] = walk_forward(train, test, seasonal=(m > 1), m=m)
wf["WES"] = walk_forward_wes(train, test, m)

fc = forecast_multimodelo(series, m, freq, steps=6, rf_model=rf, xgb_model=xgb, ridge_model=ridge)

# Guardar a buffers
buf_wf, buf_fc = BytesIO(), BytesIO()
wf.to_csv(buf_wf, index=True)
fc.to_csv(buf_fc, index=True)

st.download_button("Descargar walk_forward.csv", buf_wf.getvalue(), "walk_forward.csv")
st.download_button("Descargar forecast_future.csv", buf_fc.getvalue(), "forecast_future.csv")
