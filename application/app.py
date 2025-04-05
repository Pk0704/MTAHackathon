import streamlit as st
import polars as pl
import json
from functions import show_interactive_map, figure_one, display_vehicles, figure_two, figure_three, traffic_vs_weather, cluster_labels


df=pl.read_csv('MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv')

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

# Title and description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>MTA CRZ Data Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #000000;'>Explore traffic and weather data insights for better decision-making.</p>", unsafe_allow_html=True)

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
elif visualization_option != "None":
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