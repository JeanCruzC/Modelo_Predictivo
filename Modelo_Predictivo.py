# -*- coding: utf-8 -*-
"""Forecast Multimodelo Streamlit – exporta los mismos XLSX que el notebook de Colab
Ahora permite elegir el horizonte (nº de pasos) antes de ejecutar el forecast.
Optimizado para tiempos de startup: carga y entrena modelos sólo tras subir datos y usa cache.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import BytesIO
from pmdarima import auto_arima
from pmdarima.arima import ARIMA  # fallback sencillo
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Cache para carga de datos
# ------------------------------------------------------------------
@st.cache_data
def load_data(uploaded):
    raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith(("xlsx","xls")) else pd.read_csv(uploaded)
    raw.iloc[:,0] = pd.to_datetime(raw.iloc[:,0], errors="coerce")
    raw.set_index(raw.columns[0], inplace=True)
    return raw.iloc[:,0].ffill()

# ------------------------------------------------------------------
# Funciones utilitarias
# ------------------------------------------------------------------
_arima_cache = {}

def cached_auto_arima(series, seasonal, m):
    if len(series) < 2*m:
        seasonal, m = False, 0
    key = (tuple(series.values), seasonal, m)
    if key in _arima_cache:
        return _arima_cache[key]
    try:
        model = auto_arima(series, seasonal=seasonal, m=m,
                           stepwise=True, suppress_warnings=True, random_state=42)
    except Exception:
        model = ARIMA(order=(0,0,0)).fit(series)
    _arima_cache[key] = model
    return model

@st.cache_data
def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    return 'D' if dias<=1.5 else 'W' if dias<=8 else 'MS'

@st.cache_data
def calcular_estacionalidad(series):
    n = len(series)
    if n>=24: return 12
    if n>=18: return 6
    if n>=12: return 4
    if n>=8:  return 2
    return 1

@st.cache_data
def build_features(series):
    df = pd.DataFrame({'Valor':series})
    df['Mes'] = series.index.month
    df['DiaDelAnio'] = series.index.dayofyear
    df['Lag1'] = series.shift(1)
    df['Lag2'] = series.shift(2)
    df['MediaMovil3'] = series.rolling(3).mean()
    return df.dropna()

# Walk-forward helpers

def walk_forward(train, test, seasonal, m):
    history, preds = train.copy(), []
    for idx in test.index:
        try:
            p = float(cached_auto_arima(history, seasonal, m).predict(1)[0])
        except:
            p = float(history.iloc[-1])
        preds.append(p)
        history = pd.concat([history, pd.Series(p, index=[idx])])
    return pd.Series(preds, index=test.index)


def walk_forward_wes(train, test, m):
    history, preds = train.copy(), []
    for idx in test.index:
        try:
            p = float(ExponentialSmoothing(history, trend='add', seasonal='add',
                                           seasonal_periods=m).fit().forecast(1)[0])
        except:
            p = float(history.iloc[-1])
        preds.append(p)
        history = pd.concat([history, pd.Series(p, index=[idx])])
    return pd.Series(preds, index=test.index)

# Forecast futuro
def forecast_multimodelo(series, m, freq, steps, rf, xgb):
    fechas = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq),
                           periods=steps, freq=freq)
    seasonal_flag = m>1 and len(series)>=2*m
    sarima_fc = cached_auto_arima(series, seasonal_flag, m).predict(steps).astype(float)
    arima_fc  = cached_auto_arima(series, False, 0).predict(steps).astype(float)
    try:
        wes_fc = ExponentialSmoothing(series, trend='add', seasonal='add',
                                      seasonal_periods=m).fit().forecast(steps).astype(float)
    except:
        wes_fc = np.repeat(float(series.iloc[-1]), steps)
    hist, feats = series.copy(), []
    for dt in fechas:
        l1 = float(hist.iloc[-1]); l2 = float(hist.iloc[-2] if len(hist)>1 else l1)
        mv3 = float(hist.iloc[-3:].mean() if len(hist)>=3 else l1)
        feats.append([dt.month, dt.dayofyear, l1, l2, mv3])
        hist = pd.concat([hist, pd.Series(l1, index=[dt])])
    Xf = pd.DataFrame(feats, columns=['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3'], index=fechas)
    rf_fc  = rf.predict(Xf).astype(float)
    xgb_fc = xgb.predict(Xf).astype(float)
    stack_fc = (rf_fc + xgb_fc)/2
    return pd.DataFrame({'SARIMA':sarima_fc,'ARIMA':arima_fc,'WES':wes_fc,
                         'RandomForest':rf_fc,'XGBoost':xgb_fc,'Stacking':stack_fc}, index=fechas)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.title('Forecast Multimodelo – XLSX como en Colab')

uploaded = st.file_uploader('Sube tu archivo Excel o CSV (Fecha, Valor)', type=['xlsx','xls','csv'])
steps = st.number_input('Horizonte pasos forecast', min_value=1, max_value=60, value=6)
if not uploaded:
    st.stop()

# Carga y preproceso
series = load_data(uploaded)
freq   = detectar_frecuencia(series)
m      = calcular_estacionalidad(series)
feat_df = build_features(series)

# ML training sin pasar df como parámetro
@st.cache_resource
def train_ml():
    split = int(len(feat_df)*0.8)
    Xtr = feat_df.iloc[:split][['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']]
    ytr = feat_df.iloc[:split]['Valor']
    rf = GridSearchCV(RandomForestRegressor(random_state=42),
                      {'n_estimators':[100,200],'max_depth':[5,None]}, cv=3).fit(Xtr, ytr)
    xgb = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                       {'n_estimators':[100],'learning_rate':[0.05]}, cv=3).fit(Xtr, ytr)
    return rf, xgb

rf, xgb = train_ml()

# Walk-forward
split = int(len(series)*0.8)
train, test = series.iloc[:split], series.iloc[split:]
ar_wf   = walk_forward(train, test, seasonal=False, m=0)
sr_wf   = walk_forward(train, test, seasonal=(m>1), m=m)
w_wf    = walk_forward_wes(train, test, m)
wf_df = pd.concat([ar_wf, sr_wf, w_wf], axis=1)
wf_df.columns = ['ARIMA_WFV','SARIMA_WFV','WES_WFV']

# ML test
X_test   = feat_df.iloc[split:][['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3']]
y_test   = feat_df.iloc[split:]['Valor']
rf_pred  = rf.predict(X_test).astype(float)
xgb_pred = xgb.predict(X_test).astype(float)
stack_pred= (rf_pred + xgb_pred)/2
ml_df    = pd.DataFrame({'RandomForest':rf_pred,'XGBoost':xgb_pred,'Stacking':stack_pred}, index=y_test.index)

metricas = pd.DataFrame({
    'Modelo':['ARIMA_WFV','SARIMA_WFV','WES_WFV','RandomForest','XGBoost','Stacking'],
    'RMSE':[mean_squared_error(test,ar_wf,squared=False),
            mean_squared_error(test,sr_wf,squared=False),
            mean_squared_error(test,w_wf,squared=False),
            mean_squared_error(y_test,rf_pred,squared=False),
            mean_squared_error(y_test,xgb_pred,squared=False),
            mean_squared_error(y_test,stack_pred,squared=False)]
})

# Forecast futuro
fc = forecast_multimodelo(series, m, freq, steps, rf, xgb)

# Export XLSX
buf1, buf2 = BytesIO(), BytesIO()
with pd.ExcelWriter(buf1, engine='openpyxl') as w:
    wf_df.to_excel(w, sheet_name='Modelos_Tradicionales')
    ml_df.to_excel(w, sheet_name='Modelos_ML')
    metricas.to_excel(w, sheet_name='Métricas', index=False)
with pd.ExcelWriter(buf2, engine='openpyxl') as w:
    fc.to_excel(w, sheet_name='Forecast_Multi')

# Descargas
st.download_button('Descargar predicciones_walk_forward.xlsx', buf1.getvalue(), file_name='predicciones_walk_forward.xlsx')
st.download_button('Descargar forecast_multimodelo.xlsx', buf2.getvalue(), file_name='forecast_multimodelo.xlsx')
