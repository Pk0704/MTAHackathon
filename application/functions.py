import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import seaborn as sns


from folium.plugins import MarkerCluster

def show_interactive_map(location_coords):
    """Displays an interactive folium map with markers from location_coords JSON."""
    m = folium.Map(location=[40.75, -73.97], zoom_start=12)

    for group, coords in location_coords.items():
        lat, lon = coords
        if lat is not None and lon is not None:
            folium.Marker(
                location=[lat, lon],
                popup=group,
                icon=folium.Icon(color="blue", icon="info-sign")
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

def figure_one(df):
    """
    Creates a histogram to visualize the weighted vehicle entries by hour of day.
    It groups the data based on vehicle classes and plots the distributions for cars, trucks,
    motorcycles, and taxis separately.
    """
    
    cars = df[df['Vehicle Class'] == '1 - Cars, Pickups and Vans']
    trucks = df[df['Vehicle Class'].isin(['2 - Single-Unit Trucks', '3 - Multi-Unit Trucks'])]
    motorcycles = df[df['Vehicle Class'] == '5 - Motorcycles']
    taxis = df[df['Vehicle Class'] == 'TLC Taxi/FHV']

    plt.figure(figsize=(10, 6))
    bins = range(0, 25)  # Hours 0 to 24

    plt.hist(cars['Hour of Day'], bins=bins, weights=cars['CRZ Entries'], 
            alpha=0.5, label='Cars', edgecolor='black')
    plt.hist(trucks['Hour of Day'], bins=bins, weights=trucks['CRZ Entries'], 
            alpha=0.5, label='Trucks (Lorries)', edgecolor='black')
    plt.hist(motorcycles['Hour of Day'], bins=bins, weights=motorcycles['CRZ Entries'], 
            alpha=0.5, label='Motorcycles', edgecolor='black')
    plt.hist(taxis['Hour of Day'], bins=bins, weights=taxis['CRZ Entries'], 
            alpha=0.5, label='Taxis', edgecolor='black')

    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Entries (Weighted by CRZ Entries)")
    plt.title("Weighted Vehicle Entries by Hour of Day")
    plt.legend()
    plt.xticks(bins)
    st.pyplot(plt)



# Load the data
data = pd.read_csv('MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv')
figure_one(data)
#display_vehicles(data, 'Brooklyn Bridge')