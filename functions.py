import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def plot_hourly_traffic(df):
    """Plot number of entries by hour of day."""
    hourly_counts = df.groupby("Hour of Day")["CRZ Entries"].sum()
    fig, ax = plt.subplots()
    hourly_counts.plot(kind="bar", ax=ax)
    ax.set_title("Total CRZ Entries by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("CRZ Entries")
    st.pyplot(fig)

def plot_day_of_week_traffic(df):
    """Plot number of entries by day of week."""
    day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    daily_counts = df.groupby("Day of Week")["CRZ Entries"].sum().reindex(day_order)
    fig, ax = plt.subplots()
    daily_counts.plot(kind="bar", ax=ax)
    ax.set_title("Total CRZ Entries by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("CRZ Entries")
    st.pyplot(fig)

def plot_traffic_by_detection_region(df):
    """Plot total CRZ entries by detection region."""
    region_counts = df.groupby("Detection Region")["CRZ Entries"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    region_counts.plot(kind="bar", ax=ax)
    ax.set_title("Total CRZ Entries by Detection Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("CRZ Entries")
    st.pyplot(fig)

def plot_vehicle_class_distribution(df):
    """Pie chart of vehicle class usage."""
    vehicle_counts = df["Vehicle Class"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(vehicle_counts, labels=vehicle_counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Distribution of Vehicle Classes")
    st.pyplot(fig)