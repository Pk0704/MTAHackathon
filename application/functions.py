import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import seaborn as sns


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
    
    # Display the filtered data
    st.write("Distributions: ")
    
    # Get the distribution of vehicle classes in descending order
    vehicle_counts = df['Vehicle Class'].value_counts().sort_values(ascending=False)
    
    # Rename the index to remove numbers from the x-axis
    vehicle_counts.index = vehicle_counts.index.str.replace(r'^\d+\s*-\s*', '', regex=True)
    
    # Plot the bar chart using Seaborn
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=vehicle_counts.index, y=vehicle_counts.values, ax=ax, palette="Blues_d")
    ax.set_title("Vehicle Class Distribution", fontsize=16)
    ax.set_xlabel("Vehicle Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)
    
def display_time_series(data, location):
    
    df = data[data['Detection Group'] == location]
    
    # Display the filtered data
    st.write("Distributions: ")
    
    # Get the distribution of vehicle classes
    vehicle_counts = df['Toll Hour'].value_counts()
    
    # Display the bar chart
    st.bar_chart(vehicle_counts)



# Load the data
data = pd.read_csv('data.csv')
display_vehicles(data, 'Brooklyn Bridge')