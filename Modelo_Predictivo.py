# -*- coding: utf-8 -*-
"""Forecast Multimodelo Streamlit – exporta los mismos XLSX que el notebook de Colab"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import BytesIO
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
_arima_cache = {}

def cached_auto_arima(series, seasonal, m):
    key = (tuple(series.values), seasonal, m)
    if key not in _arima_cache:
        _arima_cache[key] = auto_arima(series, seasonal=seasonal, m=m,
                                       stepwise=True, suppress_warnings=True,
                                       random_state=42)
    return _arima_cache[key]


def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    if dias <= 1.5:
        return "D"
    if dias <= 8:
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

# Engineering features para ML
def build_features(series):
    df = pd.DataFrame(index=series.index)
    df["Valor"] = series
    df["Mes"] = df.index.month
    df["DiaDelAnio"] = df.index.dayofyear
    df["Lag1"] = df["Valor"].shift(1)
    df["Lag2"] = df["Valor"].shift(2)
    df["MediaMovil3"] = df["Valor"].rolling(3).mean()
    return df.dropna()

# Walk‑forward helpers
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
                ExponentialSmoothing(history, trend="add", seasonal="add", seasonal_periods=m)
                .fit()
                .forecast(1)[0]
            )
        except Exception:
            p = float(history.iloc[-1])
        preds.append(p)
        history = pd.concat([history, pd.Series(p, index=[test.index[t]])])
    return pd.Series(preds, index=test.index)

# Forecast futuro
def forecast_multimodelo(series, m, freq, steps, rf_model, xgb_model):
    fechas = pd.date_range(
        series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq
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

    # Features futuras para ML
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
        columns=["Mes", "DiaDelAnio", "Lag1", "Lag2", "MediaMovil3"],
        index=fechas,
    )

    rf_fc = rf_model.predict(Xf).astype(float)
    xgb_fc = xgb_model.predict(Xf).astype(float)

    stack_fc = (rf_fc + xgb_fc) / 2  # sencillo promedio

    return pd.DataFrame(
        {
            "SARIMA": sarima_fc,
            "ARIMA": arima_fc,
            "WES": wes_fc,
            "RandomForest": rf_fc,
            "XGBoost": xgb_fc,
            "Stacking": stack_fc,
        },
        index=fechas,
    )

# ------------------------------------------------------------------
# Streamlit UI mínima
# ------------------------------------------------------------------
st.title("Forecast Multimodelo – Procesamiento en Silencio (XLSX)")

uploaded = st.file_uploader("Sube tu archivo Excel/CSV (Fecha, Valor)", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.stop()

# Leer datos ------------------------------
raw = pd.read_excel(uploaded) if uploaded.name.endswith(("xlsx", "xls")) else pd.read_csv(uploaded)
raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
raw.set_index(raw.columns[0], inplace=True)
series = raw.iloc[:, 0].ffill()

freq = detectar_frecuencia(series)
m = calcular_estacionalidad(series)

# ML features -----------------------------
feat_df = build_features(series)
X_all = feat_df[["Mes", "DiaDelAnio", "Lag1", "Lag2", "MediaMovil3"]]
y_all = feat_df["Valor"]

split = int(len(X_all) * 0.8)
X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]

rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [5, None]},
    cv=3,
).fit(X_train, y_train)

xgb = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    {"n_estimators": [100], "learning_rate": [0.05]},
    cv=3,
).fit(X_train, y_train)

rf_pred = rf.predict(X_test).astype(float)
xgb_pred = xgb.predict(X_test).astype(float)
stack_pred = (rf_pred + xgb_pred) / 2

# Walk‑forward ----------------------------
train_series = series.iloc[:int(len(series) * 0.8)]
test_series = series.iloc[int(len(series) * 0.8) :]

arima_wf = walk_forward(train_series, test_series, seasonal=False, m=0)
sarima_wf = walk_forward(train_series, test_series, seasonal=(m > 1), m=m)
wes_wf = walk_forward_wes(train_series, test_series, m)

wf_df = pd.DataFrame(
    {
        "Real": test_series.values,
        "ARIMA_WFV": arima_wf.values,
        "SARIMA_WFV": sarima_wf.values,
        "WES_WFV": wes_wf.values,
    },
    index=test_series.index,
)

ml_df = pd.DataFrame(
    {
        "Real": y_test.values,
        "RandomForest": rf_pred,
        "XGBoost": xgb_pred,
        "Stacking": stack_pred,
    },
    index=y_test.index,
)

# Métricas --------------------------------
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
metricas = pd.DataFrame(
    {
        "Modelo": [
            "SARIMA_WFV",
            "ARIMA_WFV",
            "WES_WFV",
            "RandomForest",
            "XGBoost",
            "Stacking",
        ],
        "RMSE": [
            rmse(test_series, sarima_wf),
            rmse(test_series, arima_wf),
            rmse(test_series, wes_wf),
            rmse(y_test, rf_pred),
            rmse(y_test, xgb_pred),
            rmse(y_test, stack_pred),
        ],
    }
)

# ----------------------------------------
# Forecast futuro -------------------------
fc_df = forecast_multimodelo(series, m, freq, steps=6, rf_model=rf, xgb_model=xgb)

# ----------------------------------------
# Exportar XLSX como en Colab -------------
# 1) Walk‑forward + ML + métricas ---------
buf1 = BytesIO()
with pd.ExcelWriter(buf1, engine="openpyxl") as writer:
    wf_df.to_excel(writer, sheet_name="Modelos_Tradicionales")
    ml_df.to_excel(writer, sheet_name="Modelos_ML")
    metricas.to_excel(writer, sheet_name="Métricas", index=False)

# 2) Forecast futuro ----------------------
buf2 = BytesIO()
with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
    fc_df.to_excel(writer, sheet_name="Forecast_Multi")

# ----------------------------------------
# Botones de descarga ---------------------
st.download_button("Descargar predicciones_walk_forward.xlsx", buf1.getvalue(),
                   file_name="predicciones_walk_forward.xlsx")

st.download_button("Descargar forecast_multimodelo.xlsx", buf2.getvalue(),
                   file_name="forecast_multimodelo.xlsx")
