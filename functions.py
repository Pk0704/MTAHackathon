import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

def show_interactive_map(df):
    """Displays an interactive folium map with CRZ entries by location."""
    m = folium.Map(location=[40.75, -73.97], zoom_start=12)

    for _, row in df.iterrows():
        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
            popup = f"{row['Detection Group']}<br>CRZ Entries: {row['CRZ Entries']}"
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=popup,
                icon=folium.Icon(color="blue", icon="car", prefix="fa")
            ).add_to(m)

    st_folium(m, width=800, height=500)
    
def plot_traffic_by_detection_region(df):
    """Plot total CRZ entries by detection region."""
    region_counts = df.groupby("Detection Region")["CRZ Entries"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    region_counts.plot(kind="bar", ax=ax)
    ax.set_title("Total CRZ Entries by Detection Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("CRZ Entries")
    st.pyplot(fig)