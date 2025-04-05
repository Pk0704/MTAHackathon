import streamlit as st
import pandas as pd
import json
from functions import (
    # plot_traffic_by_detection_region,
    show_interactive_map
)


st.title("MTA CRZ Data Dashboard")

st.write("## Raw Data")

# Data Cleaning
df = pd.read_csv("data.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
st.dataframe(df, use_container_width=True, hide_index=True)

# # Data Visualization
# st.write("## Visualizations")
# plot_traffic_by_detection_region(df)


# Interactive Map
with open("detection_groups.json", "r") as f:
    location_coords = json.load(f)
df["Latitude"] = df["Detection Group"].map(lambda x: location_coords.get(x, (None, None))[0])
df["Longitude"] = df["Detection Group"].map(lambda x: location_coords.get(x, (None, None))[1])
df_map = df.dropna(subset=["Latitude", "Longitude"])
show_interactive_map(df_map)
