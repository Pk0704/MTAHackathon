import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os

df = pd.read_csv("data.csv")

aggregated_data = df.groupby(['Detection Group', 'Day of Week'])['Excluded Roadway Entries'].sum().reset_index()

nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
marker_cluster = MarkerCluster().add_to(nyc_map)

coordinates = {
    'Brooklyn Bridge': (40.7061, -73.9969),
    'West Side Highway at 60th St': (40.7711, -73.9882),
    'West 60th St': (40.7690, -73.9820),
    'Queensboro Bridge': (40.7570, -73.9542),
    'Queens Midtown Tunnel': (40.7440, -73.9712)
}

for _, row in aggregated_data.iterrows():
    loc = coordinates.get(row['Detection Group'])
    if loc:
        folium.Marker(
            location=loc,
            popup=f"{row['Detection Group']}<br>Day: {row['Day of Week']}<br>Excluded Entries: {row['Excluded Roadway Entries']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

map_path = "nyc_traffic_map.html"
nyc_map.save(map_path)

webbrowser.open("file://" + os.path.realpath(map_path))
