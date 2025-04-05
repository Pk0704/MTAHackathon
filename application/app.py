import streamlit as st
import polars as pl
import json
from functions import show_interactive_map, figure_one, display_vehicles, figure_two, figure_three, traffic_vs_weather, cluster_labels

# Title
st.title("MTA CRZ Data Dashboard")

# Raw Data Section
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
temp = df.head(25).to_pandas()
st.dataframe(temp, use_container_width=True, hide_index=True)

# Visualizations Section
st.write("## Visualizations")
visualization_option = st.selectbox(
    "Select a visualization to display:",
    ["Vehicle Time Series Analysis", "Vehicle Class Breakdown", "Traffic Time Series Analysis", "Traffic vs Temperature", "Clustering Labels"]
)

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
        
@st.cache_data
def load_location_coords():
    with open("detection_groups.json", "r") as f:
        location_coords = json.load(f)
    # Convert to Polars DataFrame
    location_df = pl.DataFrame(
        {
            "Detection Group": list(location_coords.keys()),
            "Latitude": [v[0] for v in location_coords.values()],
            "Longitude": [v[1] for v in location_coords.values()],
        }
    )
    return location_df

@st.cache_data
def preprocess_map_data(df, location_df):
    # Check if df is already a Polars DataFrame
    if not isinstance(df, pl.DataFrame):
        # Convert Pandas DataFrame to Polars DataFrame
        df = pl.from_pandas(df)
    
    # Merge with location data
    df_merged = df.join(location_df, on="Detection Group", how="left")
    
    # Drop rows with missing Latitude/Longitude
    df_map = df_merged.filter(df_merged["Latitude"].is_not_null() & df_merged["Longitude"].is_not_null())
    return df_map

# Interactive Map Section
st.write("## Interactive Map")
with st.container():
    # Load location coordinates
    location_df = load_location_coords()
    
    # Preprocess map data
    df_map = preprocess_map_data(df, location_df)
    
    # Convert back to Pandas for Streamlit map rendering
    df_map_pandas = df_map.to_pandas()
    
    # Sample data for rendering to improve performance
    df_map_pandas_sampled = df_map_pandas.sample(n=5000, random_state=42)  # Limit to 5000 points
    
    # Display the map
    show_interactive_map(df_map_pandas_sampled)  # Display the map