import streamlit as st
import pandas as pd
from functions import (
    plot_traffic_by_detection_region,
    show_interactive_map
)


st.title("MTA CRZ Data Dashboard")

st.write("## Raw Data")
# Data Cleaning
df = pd.read_csv("data.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
st.dataframe(df, use_container_width=True, hide_index=True)

# Data Visualization
st.write("## Visualizations")
plot_traffic_by_detection_region(df)

# Add coordinates to your DataFrame
location_coords = {
    "Brooklyn Bridge": (40.7061, -73.9969),
    "West Side Highway at 60th St": (40.7721, -73.9896),
    "West 60th St": (40.7700, -73.9840),
    "Queensboro Bridge": (40.7577, -73.9580),
    "Queens Midtown Tunnel": (40.7440, -73.9712)
}

df["Latitude"] = df["Detection Group"].map(lambda x: location_coords.get(x, (None, None))[0])
df["Longitude"] = df["Detection Group"].map(lambda x: location_coords.get(x, (None, None))[1])
df_map = df.dropna(subset=["Latitude", "Longitude"])

# Display the interactive map
st.write("## Interactive Map of CRZ Traffic Locations")
show_interactive_map(df_map)
