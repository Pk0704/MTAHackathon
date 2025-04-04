import streamlit as st
import pandas as pd
from functions import (
    plot_hourly_traffic,
    plot_day_of_week_traffic,
    plot_traffic_by_detection_region,
    plot_vehicle_class_distribution,
)


st.title("MTA CRZ Data Dashboard")

# Data Cleaning
df = pd.read_csv("data.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
st.dataframe(df, use_container_width=True, hide_index=True)

# Data Vis
st.write("## Raw Data")
st.dataframe(df)

st.write("## Visualizations")
plot_hourly_traffic(df)
plot_day_of_week_traffic(df)
plot_traffic_by_detection_region(df)
plot_vehicle_class_distribution(df)