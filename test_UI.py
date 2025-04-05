import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import datetime

# ---- SETUP ----
st.set_page_config(layout="wide")
st.title("NYC Congestion Relief Zone — Full Interactive Dashboard")

# Load CSV
df = pd.read_csv("data.csv")

# Convert time-related fields
datetime_cols = ["Toll Date", "Toll Hour", "Toll 10 Minute Block"]
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Identify columns
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Add Day Type (Weekday/Weekend)
if "Day of Week" in df.columns and "Day Type" not in df.columns:
    df["Day Type"] = df["Day of Week"].apply(lambda x: "Weekend" if x in ["Saturday", "Sunday"] else "Weekday")

# ------------------ SIDEBAR FILTERS ------------------ #
st.sidebar.header("Filter the Dataset")

# Dynamic categorical filters
filter_values = {}
for col in categorical_cols:
    unique_vals = sorted(df[col].dropna().unique())
    selected_val = st.sidebar.selectbox(f"{col}", ["All"] + unique_vals, key=f"filter_{col}")
    if selected_val != "All":
        filter_values[col] = selected_val

# Hour slider
if "Hour of Day" in df.columns:
    hour_range = st.sidebar.slider("Hour Range", min_value=0, max_value=23, value=(0, 23))

# Date range
if "Toll Date" in df.columns:
    min_date = df["Toll Date"].min()
    max_date = df["Toll Date"].max()
    selected_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Map controls
max_points = st.sidebar.slider("Max Points on Map", 10, 1000, 200)
show_map = st.sidebar.checkbox("Show Map", value=True)

# ------------------ APPLY FILTERS ------------------ #
filtered_df = df.copy()

# Apply category filters
for col, val in filter_values.items():
    filtered_df = filtered_df[filtered_df[col] == val]

# Apply hour range
if "Hour of Day" in df.columns:
    filtered_df = filtered_df[filtered_df["Hour of Day"].between(hour_range[0], hour_range[1])]

# Apply date range
if "Toll Date" in df.columns and len(selected_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Toll Date"] >= pd.to_datetime(selected_range[0])) &
        (filtered_df["Toll Date"] <= pd.to_datetime(selected_range[1]))
    ]

# ------------------ CHART TABS ------------------ #
daily_tab, hourly_tab = st.tabs(["Daily", "Hourly"])

with daily_tab:
    st.subheader("Daily Vehicle Entries")

    if "CRZ Entries" in filtered_df.columns:
        daily_data = filtered_df.groupby(["Toll Date", "Day Type"]).agg({
            "CRZ Entries": "sum"
        }).reset_index()

        fig_daily = px.bar(
            daily_data,
            x="Toll Date",
            y="CRZ Entries",
            color="Day Type",
            barmode="group",
            title="Daily Vehicle Entries to the CRZ",
            labels={"CRZ Entries": "Total Entries", "Toll Date": "Date"},
        )
        fig_daily.update_layout(margin=dict(l=10, r=10, t=40, b=30), height=500)
        st.plotly_chart(fig_daily, use_container_width=True)

with hourly_tab:
    st.subheader("Hourly Vehicle Entry Distribution")

    if "Hour of Day" in filtered_df.columns and "CRZ Entries" in filtered_df.columns:
        hourly_data = filtered_df.groupby(["Hour of Day", "Day Type"]).agg({
            "CRZ Entries": "sum"
        }).reset_index()

        fig_hourly = px.bar(
            hourly_data,
            x="Hour of Day",
            y="CRZ Entries",
            color="Day Type",
            barmode="group",
            title="Vehicle Entries by Hour",
            labels={"CRZ Entries": "Total Entries", "Hour of Day": "Hour"},
        )
        fig_hourly.update_layout(margin=dict(l=10, r=10, t=40, b=30), height=500)
        st.plotly_chart(fig_hourly, use_container_width=True)

# ------------------ MAP ------------------ #
if show_map:
    st.subheader("Detection Points Map")

    # Coordinates for key detection points
    coordinates = {
        'Brooklyn Bridge': (40.7061, -73.9969),
        'West Side Highway at 60th St': (40.7711, -73.9882),
        'West 60th St': (40.7690, -73.9820),
        'Queensboro Bridge': (40.7570, -73.9542),
        'Queens Midtown Tunnel': (40.7440, -73.9712)
    }

    # Reduce size for performance
    map_df = filtered_df.head(max_points)

    # Default to midpoint if nothing valid
    m = folium.Map(location=[40.75, -73.97], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in map_df.iterrows():
        location = coordinates.get(row.get("Detection Group"))
        if location and pd.notna(row.get("CRZ Entries")):
            entries = row.get("CRZ Entries", 0)
            excluded = row.get("Excluded Roadway Entries", 0)
            vehicle_class = row.get("Vehicle Class", "Unknown")
            detection_region = row.get("Detection Region", "Unknown")
            time = row.get("Toll Hour", "Unknown")
            weekday = row.get("Day of Week", "Unknown")

            popup_html = f"""
            <div style="font-size: 14px;">
                <b>Location:</b> {row['Detection Group']}<br>
                <b>Region:</b> {detection_region}<br>
                <b>Vehicle Class:</b> {vehicle_class}<br>
                <b>Date/Time:</b> {time}<br>
                <b>Day:</b> {weekday}<br>
                <b>CRZ Entries:</b> {entries:,}<br>
                <b>Excluded Entries:</b> {excluded:,}
            </div>
            """

            folium.CircleMarker(
                location=location,
                radius=min(max(entries / 20, 4), 15),  # scale radius
                fill=True,
                fill_opacity=0.7,
                color=None,
                fill_color="crimson" if entries > 150 else "orange" if entries > 75 else "green",
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['Detection Group']} ({entries} entries)"
            ).add_to(marker_cluster)

    st_folium(m, width=1000, height=600)
