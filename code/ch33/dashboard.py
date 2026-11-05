# The demand-forecasting dashboard (run: streamlit run dashboard.py).
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from demand_data import windows, TIME, df, TEST
from forecasters import GRUForecaster, MLPForecaster

st.set_page_config(page_title="Bike demand forecast", layout="wide")


@st.cache_resource
def load_models():
    models = {"GRU": GRUForecaster(), "MLP": MLPForecaster()}
    for name, m in models.items():
        m.load_state_dict(torch.load(f"{name.lower()}_forecaster.pt")); m.eval()
    return models


models = load_models()
P, F, Y, origins = windows(*TEST, step=24)
days = [TIME[o].date() for o in origins]
st.sidebar.title("Bike demand")
day = st.sidebar.selectbox("Forecast for", days, index=days.index(pd.Timestamp("2012-12-12").date()))
name = st.sidebar.radio("Model", list(models))
rain = st.sidebar.checkbox("What if it rains all day?")
i = days.index(day)
future = F[i:i + 1].clone()
if rain:
    future[0, :, 8] = 1.0                                   # the 'rain or snow' feature, for every hour
with torch.no_grad():
    q = torch.expm1(models[name](P[i:i + 1], future)[0]).clamp(min=0).numpy()
hours = TIME[origins[i]:origins[i] + 24]
history = df["cnt"].iloc[origins[i] - 72:origins[i]]
actual = df["cnt"].iloc[origins[i]:origins[i] + 24]

st.header(f"Rentals forecast for {day:%A, %d %B %Y}")
c1, c2, c3 = st.columns(3)
c1.metric("Expected rentals", f"{q[:, 1].sum():,.0f}", f"{q[:, 1].sum() - history[-24:].sum():+,.0f} vs. yesterday")
c2.metric("Busiest hour", f"{hours[q[:, 1].argmax()]:%H:00}", f"{q[:, 1].max():,.0f} rentals")
peak = q[:, 1].argmax()
c3.metric("Busiest hour, 80% range", f"{q[peak, 0]:,.0f} to {q[peak, 2]:,.0f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=history.index, y=history, name="past 3 days", line=dict(color="#5A6478")))
fig.add_trace(go.Scatter(x=list(hours) + list(hours[::-1]), y=list(q[:, 2]) + list(q[::-1, 0]), fill="toself",
                         fillcolor="rgba(11,132,165,0.2)", line=dict(width=0), name="80% interval"))
fig.add_trace(go.Scatter(x=hours, y=q[:, 1], name=f"{name} forecast", line=dict(color="#0B84A5", width=3)))
if not rain:
    fig.add_trace(go.Scatter(x=hours, y=actual, name="what happened", line=dict(color="#D6339F", dash="dot")))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="rentals per hour",
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)
st.caption("Weather inputs are the recorded weather for the day, standing in for a weather forecast.")
