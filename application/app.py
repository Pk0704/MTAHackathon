import folium
import pandas as pd
import streamlit as st
import polars as pl
import json
from functions import load_location_coords, preprocess_map_data, generate_interactive_map, figure_one, display_vehicles, figure_two, figure_three, traffic_vs_weather, cluster_labels
import pydeck as pdk
import streamlit_folium as st_folium 
from streamlit_folium import folium_static


df = pl.read_csv('MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv')

# Set page configuration
st.set_page_config(
    page_title="MTA CRZ Data Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* Change the background color */
    .css-18e3th9 {
        background-color: #F0F2F6;  /* Light gray background */
    }
    /* Change the sidebar background color */
    .css-1d391kg {
        background-color: #FFFFFF;  /* White sidebar */
    }
    /* Change the primary color (buttons, sliders, etc.) */
    .css-1cpxqw2 {
        color: #4CAF50 !important;  /* Green primary color */
    }
    /* Change the font color */
    .css-10trblm {
        color: #000000;  /* Black text */
    }
    /* Change the font family */
    html, body, [class*="css"] {
        font-family: 'Georgia', serif !important;  /* Use Georgia font globally */
    }
    h1, h2, h3, h4, h5, h6, p, div {
        font-family: 'Georgia', serif !important;  /* Ensure headers and paragraphs use Georgia */
    }
    /* Add padding to the main content */
    .block-container {
        padding: 2rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description with MTA logo
st.image("mta_logo.png", width=120)  # Display the local MTA logo
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #4CAF50;'>MTA CRZ Data Dashboard</h1>
        <p style='color: #FFFFFF;'>Explore traffic and weather data insights for better decision-making.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for navigation
st.sidebar.title("Navigation")

# Section 1: Visualization Selector
st.sidebar.subheader("Visualizations")
visualization_option = st.sidebar.selectbox(
    "Select a visualization to display:",
    ["Raw Data", "Vehicle Time Series Analysis", "Vehicle Class Breakdown", "Traffic Time Series Analysis", "Traffic vs Temperature", "Clustering Labels"]
)

# Section 2: Interactive Map
st.sidebar.subheader("Interactive Map")
show_map = st.sidebar.checkbox("Show Interactive Map")

# Main Content: Raw Data
if visualization_option == "Raw Data" and not show_map:
    st.write("## Raw Data")
    @st.cache_data
    def load_data():
        # Load the CSV file into a Polars DataFrame
        df = pl.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
        # Drop the "Unnamed: 0" column if it exists
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0")
        return df

    df = load_data()
    # Convert to Pandas for Streamlit's dataframe rendering
    st.dataframe(df, use_container_width=True, hide_index=True)

# Visualizations Section
elif visualization_option != "None" and not show_map:
    st.write("## Visualizations")
    # Create a placeholder for the visualization
    visualization_placeholder = st.empty()

    # Clear the placeholder and display the selected visualization
    if visualization_option == "Vehicle Time Series Analysis":
        visualization_placeholder.empty()  # Clear the placeholder
        with visualization_placeholder.container():
            figure_one(df.to_pandas())  # Convert to Pandas if the visualization function requires it
    elif visualization_option == "Vehicle Class Breakdown":
        visualization_placeholder.empty()  # Clear the placeholder
        with visualization_placeholder.container():
            figure_two(df.to_pandas())  # Convert to Pandas if the visualization function requires it
    elif visualization_option == "Traffic Time Series Analysis":
        visualization_placeholder.empty()  # Clear the placeholder
        with visualization_placeholder.container():
            figure_three(df.to_pandas())  # Convert to Pandas if the visualization function requires it
    elif visualization_option == "Traffic vs Temperature":
        visualization_placeholder.empty()  # Clear the placeholder
        with visualization_placeholder.container():
            traffic_vs_weather(df.to_pandas())  # Convert to Pandas if the visualization function requires it
    elif visualization_option == "Clustering Labels":
        visualization_placeholder.empty()  # Clear the placeholder
        with visualization_placeholder.container():
            cluster_labels(df.to_pandas())  # Convert to Pandas if the visualization function requires it

# Interactive Map Section
elif show_map:
    st.write("## Interactive Map")
    with st.container():
        location_df = load_location_coords()
        if location_df is None:
            st.error("Failed to load location coordinates.")
        else:
            df_map = preprocess_map_data(df, location_df)
            if df_map is None or df_map.is_empty():
                st.error("No valid data available for the map.")
            else:
                df_map_pandas = df_map.to_pandas()

                aggregated_data = (
                    df_map_pandas.groupby(["Latitude", "Longitude", "Detection Group"])["CRZ Entries"]
                    .sum()
                    .reset_index()
                )
                col1, _ = st.columns([1, 4])  # Left column is smaller
                with col1:
                    style_choice = st.selectbox("Choose map style:", ["Satellite", "Standard", "Light","Dark"])
                # Generate and display the map
                m = generate_interactive_map(aggregated_data, style=style_choice)
                folium_static(m)