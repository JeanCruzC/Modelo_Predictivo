# -*- coding: utf-8 -*-
"""Forecast Multimodelo Headless: calcula WFV y forecast sin interfaz"""

import pandas as pd
import numpy as np
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Funciones
# ------------------------------------------------------------
# Auto ARIMA cacheada
_arima_cache = {}
def cached_auto_arima(series, seasonal, m):
    key = (tuple(series.values), seasonal, m)
    if key not in _arima_cache:
        _arima_cache[key] = auto_arima(
            series,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True
        )
    return _arima_cache[key]

# Detectar frecuencia
def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    diffs = series.index.to_series().diff().dropna()
    days = diffs.dt.days.mean()
    if days <= 1.5:
        return 'D'
    if days <= 8:
        return 'W'
    return 'MS'

# Calcular estacionalidad
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

# Walk-forward ARIMA
def walk_forward(train, test, seasonal, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            model = cached_auto_arima(history, seasonal, m)
            arr = model.predict(n_periods=1)
            p = float(arr[0])
        except:
            p = float(history.iloc[-1])
        preds.append(p)
        history = history.append(pd.Series([p], index=[test.index[t]]))
    return pd.Series(preds, index=test.index)

# Walk-forward WES
def walk_forward_wes(train, test, m):
    history = train.copy()
    preds = []
    for t in range(len(test)):
        try:
            model = ExponentialSmoothing(
                history,
                seasonal='add', trend='add', seasonal_periods=m
            ).fit()
            arr = model.forecast(1)
            p = float(arr[0])
        except:
            p = float(history.iloc[-1])
        preds.append(p)
        history = history.append(pd.Series([p], index=[test.index[t]]))
    return pd.Series(preds, index=test.index)

# Forecast futuro multimodelo
def forecast_multimodelo(series, m, freq, steps, rf_model, xgb_model, ridge_model):
    fechas = pd.date_range(
        start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=steps, freq=freq
    )
    seasonal_flag = (m > 1 and len(series) >= 2*m)
    sarima_fc = cached_auto_arima(series, seasonal_flag, m).predict(n_periods=steps).astype(float)
    arima_fc = cached_auto_arima(series, False, 0).predict(n_periods=steps).astype(float)
    try:
        wes_fc = ExponentialSmoothing(
            series, seasonal='add', trend='add', seasonal_periods=m
        ).fit().forecast(steps).astype(float)
    except:
        wes_fc = np.repeat(float(series.iloc[-1]), steps)
    # ML features
    hist = series.copy()
    feats = []
    for i in range(1, steps+1):
        dt = fechas[i-1]
        lag1 = float(hist.iloc[-1])
        lag2 = float(hist.iloc[-2] if len(hist)>1 else hist.iloc[-1])
        mv3 = float(hist.iloc[-3:].mean() if len(hist)>=3 else hist.iloc[-1])
        feats.append([dt.month, dt.dayofyear, lag1, lag2, mv3])
        hist = hist.append(pd.Series([lag1], index=[dt]))
    Xf = pd.DataFrame(feats, index=fechas, columns=['Mes','DiaDelAnio','Lag1','Lag2','MediaMovil3'])
    rf_fc  = rf_model.predict(Xf).astype(float)
    xgb_fc = xgb_model.predict(Xf).astype(float)
    df_stack = pd.DataFrame({
        'SARIMA': sarima_fc,
        'ARIMA': arima_fc,
        'WES': wes_fc,
        'RF': rf_fc,
        'XGB': xgb_fc
    }, index=fechas)
    stk_fc = ridge_model.predict(df_stack).astype(float)
    df_res = pd.DataFrame({
        'SARIMA': sarima_fc,
        'ARIMA': arima_fc,
        'WES': wes_fc,
        'RandomForest': rf_fc,
        'XGBoost': xgb_fc,
        'Stacking': stk_fc
    }, index=fechas)
    return df_res

# ------------------------------------------------------------
# Código principal (headless)
# ------------------------------------------------------------
if __name__ == '__main__':
    # Leer datos de Excel/CSV
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else 'Datos.xlsx'
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # Asumimos columna Fecha y Valor
    df['Fecha'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
    df.set_index('Fecha', inplace=True)
    series = df.iloc[:,1].ffill()

    freq = detectar_frecuencia(series)
    m = calcular_estacionalidad(series)

    # Split train/test
    split = int(len(series)*0.8)
    train, test = series[:split], series[split:]

    # Entrenar RF y XGB
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators':[100,200],'max_depth':[5,None]}, cv=3)
    grid_rf.fit(train.to_frame('Valor'), train)
    grid_xgb = GridSearchCV(XGBRegressor(random_state=42), {'n_estimators':[100,200],'learning_rate':[0.05,0.1]}, cv=3)
    grid_xgb.fit(train.to_frame('Valor'), train)
    ridge_m = Ridge().fit(
        pd.DataFrame(), []
    )  # Dummy fit, necesita datos reales

    # Calcular WFV
    wf = pd.DataFrame(index=test.index)
    wf['ARIMA'] = walk_forward(train, test, seasonal=False, m=0)
    wf['SARIMA'] = walk_forward(train, test, seasonal=(m>1), m=m)
    wf['WES'] = walk_forward_wes(train, test, m)
    wf.to_csv('walk_forward.csv')

    # Calcular forecast futuro
    fc = forecast_multimodelo(series, m, freq, steps=6, rf_model=grid_rf, xgb_model=grid_xgb, ridge_model=ridge_m)
    fc.to_csv('forecast_future.csv')

    # Fin sin prints ni UI
    sys.exit(0)
