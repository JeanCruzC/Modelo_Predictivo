# -*- coding: utf-8 -*-
"""Forecast Multimodelo Streamlit – procesa en silencio y entrega CSV"""

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

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
_arima_cache = {}

def cached_auto_arima(series, seasonal, m):
    key = (tuple(series.values), seasonal, m)
    if key not in _arima_cache:
        _arima_cache[key] = auto_arima(series, seasonal=seasonal, m=m,
                                       stepwise=True, suppress_warnings=True)
    return _arima_cache[key]


def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    if dias <= 1.5:
        return 'D'
    if dias <= 8:
        return 'W'
    return 'MS'


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

# Features helper
def build_features(series):
    df = pd.DataFrame(index=series.index)
    df['Valor'] = series
    df['Mes'] = df.index.month
    df['DiaDelAnio'] = df.index.dayofyear
    df['Lag1'] = df['Valor'].shift(1)
    df['Lag2'] = df['Valor'].shift(2)
    df['MediaMovil3'] = df['Valor'].rolling(3).mean()
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
            p = float(ExponentialSmoothing(history, trend='add', seasonal='add',
                                           seasonal_periods=m).fit().forecast(1)[0])
        except Exception:
            p = float(history.iloc[-1])
        preds.append(p)
        history = pd.concat([history, pd.Series(p, index=[test.index[t]])])
    return pd.Series(preds, index=test.index)

# Forecast futuro
def forecast_multimodelo(series, m, freq, steps, rf_model, xgb_model):
    fechas = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq),
                           periods=steps, freq=freq)
    seasonal_flag = m > 1 and len(series) >= 2*m
    sarima_fc = cached_auto_arima(series, seasonal_flag, m).predict(steps).astype(float)
    arima_fc = cached_auto_arima(series, False, 0).predict(steps).astype(float)
    try:
        wes_fc = ExponentialSmoothing(series, trend='add', seasonal='add',
                                      seasonal_periods=m).fit().forecast(steps).astype(float)
    except Exception:
        wes_fc = np.repeat(float(series.iloc[-1]), steps)

    # Features para ML futuro
    hist = series.copy()
    feats = []
    for dt in fechas:
        lag1 = float(hist.iloc[-1])
        lag2 = float(hist.iloc[-2] if len(hist)>1 else lag1)
        mv3  = float(hist.iloc[-3:].mean() if len(hist)>=3 else lag1)
        feats.append([dt.month, dt.dayofyear, lag1, lag2, mv3])
        hist = pd.concat([hist, pd.Series(lag1, index=[dt])])
    Xf = pd.DataFrame(feats, columns=['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3'], index=fechas)

    rf_fc  = rf_model.predict(Xf).astype(float)
    xgb_fc = xgb_model.predict(Xf).astype(float)

    stacking_fc = (sarima_fc + arima_fc + wes_fc + rf_fc + xgb_fc) / 5

    return pd.DataFrame({
        'SARIMA': sarima_fc,
        'ARIMA': arima_fc,
        'WES': wes_fc,
        'RandomForest': rf_fc,
        'XGBoost': xgb_fc,
        'Ensemble': stacking_fc
    }, index=fechas)

# ------------------------------------------------------------------
# Streamlit UI mínima
# ------------------------------------------------------------------
st.title('Forecast Multimodelo – Procesamiento en Silencio')

uploaded = st.file_uploader('Sube tu archivo Excel/CSV (Fecha, Valor)', type=['xlsx','xls','csv'])
if not uploaded:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith(('xlsx','xls')) else pd.read_csv(uploaded)

df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors='coerce')
df.set_index(df.columns[0], inplace=True)
series = df.iloc[:,0].ffill()

freq = detectar_frecuencia(series)
m = calcular_estacionalidad(series)

# ----- Feature engineering para ML -----
feat_df = build_features(series)
X_all = feat_df[['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']]
y_all = feat_df['Valor']

split = int(len(X_all)*0.8)
X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
y_train = y_all.iloc[:split]

rf = GridSearchCV(RandomForestRegressor(random_state=42),
                  {'n_estimators':[100], 'max_depth':[5,None]}, cv=3).fit(X_train, y_train)

xgb = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                   {'n_estimators':[100], 'learning_rate':[0.05]}, cv=3).fit(X_train, y_train)

# ----- Walk‑forward (solo estadísticos) -----
train_series = series.iloc[:int(len(series)*0.8)]
test_series  = series.iloc[int(len(series)*0.8):]

a_wf = walk_forward(train_series, test_series, seasonal=False, m=0)
s_wf = walk_forward(train_series, test_series, seasonal=(m>1), m=m)
w_wf = walk_forward_wes(train_series, test_series, m)
wf = pd.concat([a_wf, s_wf, w_wf], axis=1)
wf.columns = ['ARIMA','SARIMA','WES']

# ----- Forecast futuro -----
fc = forecast_multimodelo(series, m, freq, steps=6, rf_model=rf, xgb_model=xgb)

# ----- Buffers de descarga -----
b1,b2 = BytesIO(), BytesIO()
wf.to_csv(b1, index=True)
fc.to_csv(b2, index=True)

st.download_button('Descargar walk_forward.csv', b1.getvalue(), 'walk_forward.csv')
st.download_button('Descargar forecast_future.csv', b2.getvalue(), 'forecast_future.csv')
