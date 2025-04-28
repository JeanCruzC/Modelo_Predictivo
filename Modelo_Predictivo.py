# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# === Funciones utilitarias ===
_arima_cache = {}
def cached_auto_arima(series, seasonal, m):
    key = (tuple(series.values), seasonal, m)
    if key not in _arima_cache:
        _arima_cache[key] = auto_arima(series, seasonal=seasonal, m=m, stepwise=True, suppress_warnings=True, random_state=42)
    return _arima_cache[key]

def detectar_frecuencia(series):
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    if dias <= 1.5: return "D"
    if dias <= 8:   return "W"
    return "MS"

def calcular_estacionalidad(series):
    n = len(series)
    if n >= 24: return 12
    if n >= 18: return 6
    if n >= 12: return 4
    if n >= 8:  return 2
    return 1

def build_features(series):
    df = pd.DataFrame(index=series.index)
    df["Valor"] = series
    df["Mes"] = df.index.month
    df["DiaDelAnio"] = df.index.dayofyear
    df["Lag1"] = df["Valor"].shift(1)
    df["Lag2"] = df["Valor"].shift(2)
    df["MediaMovil3"] = df["Valor"].rolling(3).mean()
    return df.dropna()

# === Streamlit UI ===
st.title("Forecast Multimodelo Rápido")
uploaded_file = st.file_uploader("Carga el archivo Excel o CSV (Fecha, Valor)", type=["xlsx", "xls", "csv"])

if uploaded_file:
    st.write("Archivo cargado correctamente.")

    raw = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file)
    raw.iloc[:,0] = pd.to_datetime(raw.iloc[:,0], format="%d/%m/%Y", errors="coerce")
    raw.set_index(raw.columns[0], inplace=True)
    series = raw.iloc[:,0].ffill()

    st.write(f"Datos cargados: {len(series)} registros.")

    steps_horizon = st.number_input("¿Cuántos pasos futuros deseas pronosticar?", min_value=1, max_value=200, value=6)

    if st.button("Ejecutar Pronóstico"):
        st.write("Iniciando detección de frecuencia y estacionalidad...")
        freq = detectar_frecuencia(series)
        m = calcular_estacionalidad(series)
        st.write(f"Frecuencia detectada: {freq}, Estacionalidad calculada: {m}")

        st.write("Construyendo características para el modelo...")
        feat_df = build_features(series)
        X_all, y_all = feat_df.drop(columns="Valor"), feat_df["Valor"]
        split = int(len(X_all)*0.8)
        X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
        y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]

        st.write("Entrenando modelos RandomForest y XGBoost...")
        rf = GridSearchCV(RandomForestRegressor(random_state=42), {"n_estimators":[100,200], "max_depth":[5,None]}, cv=3).fit(X_train, y_train)
        xgb = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), {"n_estimators":[100], "learning_rate":[0.05]}, cv=3).fit(X_train, y_train)
        st.write("Modelos entrenados.")

        fechas = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=steps_horizon, freq=freq)
        seasonal_flag = m > 1 and len(series) >= 2*m

        st.write("Generando pronósticos...")
        sarima_fc = cached_auto_arima(series, seasonal_flag, m).predict(steps_horizon).astype(float)
        arima_fc = cached_auto_arima(series, False, 0).predict(steps_horizon).astype(float)
        wes_fc = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=m).fit().forecast(steps_horizon).astype(float)

        hist, feats = series.copy(), []
        for dt in fechas:
            lag1 = float(hist.iloc[-1])
            lag2 = float(hist.iloc[-2] if len(hist)>1 else lag1)
            mv3  = float(hist.iloc[-3:].mean() if len(hist)>=3 else lag1)
            feats.append([dt.month, dt.dayofyear, lag1, lag2, mv3])
            hist = pd.concat([hist, pd.Series(lag1, index=[dt])])

        Xf = pd.DataFrame(feats, columns=["Mes","DiaDelAnio","Lag1","Lag2","MediaMovil3"], index=fechas)

        rf_fc = rf.predict(Xf).astype(float)
        xgb_fc = xgb.predict(Xf).astype(float)
        stack_fc = (rf_fc + xgb_fc) / 2

        fc_df = pd.DataFrame({"SARIMA": sarima_fc,"ARIMA": arima_fc,"WES": wes_fc,"RandomForest": rf_fc,"XGBoost": xgb_fc,"Stacking": stack_fc}, index=fechas)

        st.write("### Predicciones")
        st.dataframe(fc_df)

        buf = BytesIO()
        with pd.ExcelWriter(buf) as writer:
            fc_df.to_excel(writer, "Forecast")
        st.download_button("Descargar Resultados.xlsx", buf.getvalue(), "Resultados.xlsx")
