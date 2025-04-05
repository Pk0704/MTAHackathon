import streamlit as st
import pandas as pd
import json
from functions import show_interactive_map, figure_one, display_vehicles

# Title
st.title("MTA CRZ Data Dashboard")

# Raw Data Section
st.write("## Raw Data")
df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
temp = df.head(25)
st.dataframe(temp, use_container_width=True, hide_index=True)

# Visualizations Section
st.write("## Visualizations")
with st.container():
    figure_one(df)  # Display the histogram

# Interactive Map Section
st.write("## Interactive Map")
with st.container():
    with open("detection_groups.json", "r") as f:
        location_coords = json.load(f)
    df["Latitude"] = df["Detection Group"].map(lambda x: location_coords.get(x, (None, None))[0])
    df["Longitude"] = df["Detection Group"].map(lambda x: location_coords.get(x, (None, None))[1])
    df_map = df.dropna(subset=["Latitude", "Longitude"])
    show_interactive_map(df_map)  # Display the map