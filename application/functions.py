from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import seaborn as sns
import requests


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

def figure_two(df):
    grouped = df.groupby(["Time Period", "Vehicle Class"])["CRZ Entries"].sum().reset_index()
    # Map the detailed vehicle classes to contracted names.
    vehicle_class_map = {
        "TLC Taxi/FHV": "TLC Taxi/FHV",
        "5 - Motorcycles": "motorcycles",
        "4 - Buses": "buses",
        "3 - Multi-Unit Trucks": "Trucks (multi)",
        "2 - Single-Unit Trucks": "trucks (Single)",
        "1 - Cars, Pickups and Vans": "cars, pickups, vans"
    }
    grouped["Vehicle Class"] = grouped["Vehicle Class"].map(vehicle_class_map)
    pivot = grouped.pivot(index="Vehicle Class", columns="Time Period", values="CRZ Entries").fillna(0)
    # For each vehicle class, compute the proportion for each time period (so they add to 1).
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)

    # Plot a grouped bar chart: vehicle class on x-axis, two bars per class (Peak and Overnight)
    plt.figure(figsize=(10, 6))
    # We'll use a custom color set (e.g., yellow and blue)
    colors = {"Peak": "Crimson", "Overnight": "blue"}
    pivot_prop.plot(kind="bar", stacked=False, color=[colors.get(tp, "gray") for tp in pivot_prop.columns], figsize=(10,6))
    plt.xlabel("Vehicle Class", fontsize=12)
    plt.ylabel("Proportion of Total CRZ Entries", fontsize=12)
    plt.title("Proportion of CRZ Entries by Time Period for Each Vehicle Class", fontsize=14)
    plt.xticks(rotation=0)  # Rotate x-axis labels horizontally.
    plt.legend(title="Time Period", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
    
def figure_three(df):
    # --------------------------------------
    # Weather API & Visual Analysis
    # --------------------------------------
    API_KEY = "b1b230ed8664410080803403250504"
    BASE_URL = "http://api.weatherapi.com/v1/history.json"

    def fetch_weather(date_str, api_key=API_KEY, location="New York"):
        params = {
            "key": api_key,
            "q": location,
            "dt": date_str
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        forecast_day = data["forecast"]["forecastday"][0]["day"]
        return {
            "date": date_str,
            "avgtemp_c": forecast_day["avgtemp_c"],
            "avghumidity": forecast_day["avghumidity"],
            "totalprecip_mm": forecast_day["totalprecip_mm"],
            "condition": forecast_day["condition"]["text"]
        }

    df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
    print("Traffic Data sample:")
    print(df.head())

    traffic_daily = df.groupby("Toll Date")["CRZ Entries"].sum().reset_index()
    traffic_daily.rename(columns={"CRZ Entries": "daily_crz_entries"}, inplace=True)
    print("\nDaily Traffic Totals (first 5 rows):")
    print(traffic_daily.head())

    unique_dates = traffic_daily["Toll Date"].unique()
    weather_records = []
    for date in unique_dates:
        try:
            weather_data = fetch_weather(date)
            weather_records.append(weather_data)
            print(f"Fetched weather for {date}")
        except Exception as e:
            print(f"Failed to fetch weather for {date}: {e}")

    weather_df = pd.DataFrame(weather_records)
    print("\nWeather Data sample:")
    print(weather_df.head())

    merged_df = pd.merge(traffic_daily, weather_df, left_on="Toll Date", right_on="date", how="inner")

    # --- Bundle Weather Conditions into Categories ---
    weather_category_map = {
        "Sunny": "Favorable",
        "Clear": "Favorable",
        "Partly cloudy": "Favorable",
        "Overcast": "Neutral",
        "Cloudy": "Neutral",
        "Patchy rain possible": "Neutral",
        "Light rain": "Neutral",
        "Light freezing rain": "Neutral",
        "Heavy rain": "Unfavorable",
        "Heavy rain at times": "Unfavorable",
        "Moderate or heavy rain shower": "Unfavorable",
        "Moderate rain": "Unfavorable",
        "Moderate rain at times": "Unfavorable",
        "Moderate or heavy snow showers": "Unfavorable",
        "Patchy moderate snow": "Unfavorable",
        "Moderate snow": "Unfavorable"
    }
    merged_df["weather_category"] = merged_df["condition"].apply(lambda cond: weather_category_map.get(cond, "Neutral"))

    features = ["avgtemp_c", "avghumidity", "totalprecip_mm", "condition"]
    X = merged_df[features]
    y = merged_df["daily_crz_entries"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(), ["condition"])
    ], remainder="passthrough")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    pipeline.fit(X, y)

    model = pipeline.named_steps["regressor"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = dict(zip(feature_names, model.coef_))
    print("\nRegression Model Coefficients:")
    for feature, coef in coefficients.items():
        print(f"{feature}: {coef:.4f}")
    r_squared = pipeline.score(X, y)

    # ---------------------------
    # Visual Analysis: Powerful Graphs for Weather & Traffic Variability
    # ---------------------------
    merged_df['Toll Date'] = pd.to_datetime(merged_df['Toll Date'], format='%m/%d/%Y')

    # Graph 1: Daily Traffic Volume Over Time with Bundled Weather Categories
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=merged_df, x='Toll Date', y='daily_crz_entries', marker='o', color='gray', label='Daily Traffic')
    sns.scatterplot(data=merged_df, x='Toll Date', y='daily_crz_entries', hue='weather_category',
                    palette='Set1', s=100, edgecolor='black')
    plt.title('Daily Traffic Volume Over Time (Weather Categories)')
    plt.xlabel('Date')
    plt.ylabel('Daily CRZ Entries')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Weather Category')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def traffic_vs_weather(df):
    """
    Generates a scatter plot of daily traffic volume vs. average temperature,
    color-coded by weather categories.
    """
    API_KEY = "b1b230ed8664410080803403250504"
    BASE_URL = "http://api.weatherapi.com/v1/history.json"

    def fetch_weather(date_str, api_key=API_KEY, location="New York"):
        params = {
            "key": api_key,
            "q": location,
            "dt": date_str
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            forecast_day = data["forecast"]["forecastday"][0]["day"]
            return {
                "date": date_str,
                "avgtemp_c": forecast_day["avgtemp_c"],
                "avghumidity": forecast_day["avghumidity"],
                "totalprecip_mm": forecast_day["totalprecip_mm"],
                "condition": forecast_day["condition"]["text"]
            }
        else:
            return None

    # Group traffic data by date
    traffic_daily = df.groupby("Toll Date")["CRZ Entries"].sum().reset_index()
    traffic_daily.rename(columns={"CRZ Entries": "daily_crz_entries"}, inplace=True)

    # Fetch weather data for unique dates
    unique_dates = traffic_daily["Toll Date"].unique()
    weather_records = []
    for date in unique_dates:
        weather_data = fetch_weather(date)
        if weather_data:
            weather_records.append(weather_data)

    # Create weather DataFrame
    weather_df = pd.DataFrame(weather_records)

    # Merge traffic and weather data
    merged_df = pd.merge(traffic_daily, weather_df, left_on="Toll Date", right_on="date", how="inner")

    # Map weather conditions to categories
    weather_category_map = {
        "Sunny": "Favorable",
        "Clear": "Favorable",
        "Partly cloudy": "Favorable",
        "Overcast": "Neutral",
        "Cloudy": "Neutral",
        "Patchy rain possible": "Neutral",
        "Light rain": "Neutral",
        "Heavy rain": "Unfavorable",
        "Moderate rain": "Unfavorable",
    }
    merged_df["weather_category"] = merged_df["condition"].apply(lambda cond: weather_category_map.get(cond, "Neutral"))

    # Convert date column to datetime
    merged_df['Toll Date'] = pd.to_datetime(merged_df['Toll Date'], format='%m/%d/%Y')

    # Generate the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', hue='weather_category', 
                    palette='Set1', s=100, edgecolor='black')
    sns.regplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', scatter=False, 
                color='black', line_kws={'linewidth': 1.5})
    plt.title('Daily Traffic Volume vs. Average Temperature')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Daily CRZ Entries')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Weather Category')
    plt.tight_layout()
    st.pyplot(plt)
    
def cluster_labels(df):
    # Load and preprocess data
    df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
    df['Toll Date'] = pd.to_datetime(df['Toll Date'], format='%m/%d/%Y')
    traffic_daily = df.groupby("Toll Date")["CRZ Entries"].sum().reset_index()
    traffic_daily.rename(columns={"CRZ Entries": "daily_crz_entries"}, inplace=True)

    # Simulated weather data (replace with actual API data if available)
    weather_data = {
        "date": traffic_daily["Toll Date"].dt.strftime('%Y-%m-%d'),
        "avgtemp_c": [15 + i % 10 for i in range(len(traffic_daily))],  # Example temperatures
        "condition": ["Sunny" if i % 3 == 0 else "Cloudy" for i in range(len(traffic_daily))]
    }
    weather_df = pd.DataFrame(weather_data)
    weather_df["weather_category"] = weather_df["condition"].apply(
        lambda cond: "Favorable" if cond == "Sunny" else "Neutral"
    )

    # Merge traffic and weather data
    merged_df = pd.merge(traffic_daily, weather_df, left_on="Toll Date", right_on="date", how="inner")

    # Plot: Traffic Volume vs. Average Temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', hue='weather_category', 
                    palette='Set1', s=100, edgecolor='black')
    sns.regplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', scatter=False, 
                color='black', line_kws={'linewidth': 1.5})
    plt.title('Daily Traffic Volume vs. Average Temperature')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Daily CRZ Entries')
    plt.legend(title='Weather Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)