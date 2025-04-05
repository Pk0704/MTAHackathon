import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import perspective
import streamlit.components.v1 as components
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

def display_vehicles(data, location):
    # Filter data by location
    df = data[data['Detection Group'] == location]
    
    # Create a Perspective table using the factory function
    table = Table(df.to_dict(orient="records"))

    # Generate HTML/JS for the Perspective viewer
    html_template = """
    <link href='https://unpkg.com/@finos/perspective-viewer/dist/css/material.css' rel='stylesheet'>
    <script src='https://unpkg.com/@finos/perspective-viewer/dist/umd/perspective-viewer.js'></script>
    <perspective-viewer style="height: 500px; width: 100%;" id="viewer"></perspective-viewer>
    <script>
      const viewer = document.getElementById("viewer");
      const table = perspective.worker().table(%s);
      viewer.load(table);
      viewer.restore({group_by: ["Vehicle Class"], columns: ["Count"]});
    </script>
    """ % table.to_arrow().to_pybytes().hex()

    # Render in Streamlit
    components.html(html_template, height=600)
    
def display_time_series(data, location):
    
    df = data[data['Detection Group'] == location]
    
    # Display the filtered data
    st.write("Distributions: ")
    
    # Get the distribution of vehicle classes
    vehicle_counts = df['Toll Hour'].value_counts()
    
    # Display the bar chart
    st.bar_chart(vehicle_counts)
    
data=pd.read_csv('data.csv')
display_vehicles(data, 'Brooklyn Bridge')