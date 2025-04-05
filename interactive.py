import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---- SETUP ----
st.set_page_config(layout="wide")
st.title("📊 Interactive Chart + 🗺️ NYC Congestion Relief Zone Map")

# Load CSV
df = pd.read_csv("data.csv")

# ------------------ SIDEBAR FILTERS ------------------ #
st.sidebar.header("📅 Filter the Dataset")

# Time Period
time_periods = sorted(df["Time Period"].dropna().unique())
selected_time = st.sidebar.selectbox("Time Period", ["All"] + time_periods)

# Day of Week
days = sorted(df["Day of Week"].dropna().unique())
selected_day = st.sidebar.selectbox("Day of Week", ["All"] + days)

# Hour of Day
hours = sorted(df["Hour of Day"].dropna().unique())
selected_hour = st.sidebar.selectbox("Hour of Day", ["All"] + list(map(str, hours)))

# Vehicle Class
vehicle_classes = sorted(df["Vehicle Class"].dropna().unique())
selected_class = st.sidebar.selectbox("Vehicle Class", ["All"] + vehicle_classes)

# Detection Group
groups = sorted(df["Detection Group"].dropna().unique())
selected_group = st.sidebar.selectbox("Detection Group", ["All"] + groups)

# Detection Region
regions = sorted(df["Detection Region"].dropna().unique())
selected_region = st.sidebar.selectbox("Detection Region", ["All"] + regions)

# Limit map size
max_points = st.sidebar.slider("Max Points on Map", min_value=10, max_value=1000, value=200)

# Show/hide map
show_map = st.sidebar.checkbox("Show Map", value=True)

# Apply filters
filtered_df = df.copy()
if selected_time != "All":
    filtered_df = filtered_df[filtered_df["Time Period"] == selected_time]
if selected_day != "All":
    filtered_df = filtered_df[filtered_df["Day of Week"] == selected_day]
if selected_hour != "All":
    filtered_df = filtered_df[filtered_df["Hour of Day"] == int(selected_hour)]
if selected_class != "All":
    filtered_df = filtered_df[filtered_df["Vehicle Class"] == selected_class]
if selected_group != "All":
    filtered_df = filtered_df[filtered_df["Detection Group"] == selected_group]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Detection Region"] == selected_region]

# ------------------ CHART ------------------ #
st.subheader("📈 Customizable Interactive Chart")

chart_type = st.selectbox("Choose chart type", ["Line", "Bar", "Scatter"])
x_axis = st.selectbox("Select X-axis", filtered_df.columns)
y_axis = st.selectbox("Select Y-axis", filtered_df.select_dtypes(include=['number']).columns)
hue = st.selectbox("Group/color by (optional)", ["None"] + list(filtered_df.columns))

# Convert date/time columns if needed
if x_axis in ["Toll Date", "Toll Hour", "Toll 10 Minute Block"]:
    filtered_df[x_axis] = pd.to_datetime(filtered_df[x_axis])

# Plotly chart
color = None if hue == "None" else hue
if chart_type == "Line":
    fig = px.line(filtered_df, x=x_axis, y=y_axis, color=color)
elif chart_type == "Bar":
    fig = px.bar(filtered_df, x=x_axis, y=y_axis, color=color)
elif chart_type == "Scatter":
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color)

fig.update_layout(
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True),
    margin=dict(l=20, r=20, t=30, b=30),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ MAP ------------------ #
if show_map:
    st.subheader("🗺️ Detection Points Map")

    coordinates = {
        'Brooklyn Bridge': (40.7061, -73.9969),
        'West Side Highway at 60th St': (40.7711, -73.9882),
        'West 60th St': (40.7690, -73.9820),
        'Queensboro Bridge': (40.7570, -73.9542),
        'Queens Midtown Tunnel': (40.7440, -73.9712)
    }

    map_df = filtered_df.head(max_points)

    m = folium.Map(location=[40.75, -73.97], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in map_df.iterrows():
        location = coordinates.get(row.get("Detection Group"))
        if location:
            popup_html = f"""
            <b>{row['Detection Group']}</b><br>
            Region: {row['Detection Region']}<br>
            Vehicle: {row['Vehicle Class']}<br>
            Time: {row['Toll Hour']}<br>
            Day: {row['Day of Week']}<br>
            Hour: {row['Hour of Day']}<br>
            Time Period: {row['Time Period']}<br>
            CRZ Entries: {row['CRZ Entries']}<br>
            Excluded Entries: {row['Excluded Roadway Entries']}
            """
            folium.CircleMarker(
                location=location,
                radius=5,
                fill=True,
                fill_opacity=0.7,
                color='blue',
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)

    st_folium(m, width=1000, height=600)
