from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium, folium_static
import seaborn as sns
import polars as pl
import json

@st.cache_data
def load_location_coords():
    """Load location coordinates from detection_groups.json."""
    try:
        with open("detection_groups.json", "r") as f:
            location_coords = json.load(f)

        location_df = pl.DataFrame(
            {
                "Detection Group": list(location_coords.keys()),
                "Latitude": [v[0] for v in location_coords.values()],
                "Longitude": [v[1] for v in location_coords.values()],
            }
        )
        return location_df
    except Exception as e:
        st.error(f"Error loading location coordinates: {e}")
        return None


@st.cache_data
def preprocess_map_data(df, location_df):
    """Merge data with location coordinates and filter valid entries."""
    try:
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        df_merged = df.join(location_df, on="Detection Group", how="left")
        df_map = df_merged.filter(
            df_merged["Latitude"].is_not_null() & df_merged["Longitude"].is_not_null()
        )
        return df_map
    except Exception as e:
        st.error(f"Error preprocessing map data: {e}")
        return None

def generate_interactive_map(aggregated_data, style='Satellite'):
    """Render a folium map with CRZ markers and customizable tile style."""

    tile_styles = {
        'Satellite': {
            'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'attr': 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, etc.'
        },
        'Light': {
            'tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            'attr': '&copy; <a href="https://carto.com/">CARTO</a>'
        },
        "Dark": {
            "tiles": "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
            "attr": "© OpenStreetMap contributors © CARTO"
        },
        'Standard': {
            'tiles': 'OpenStreetMap',
            'attr': None
        }
    }

    if style not in tile_styles:
        raise ValueError(f"Invalid map style: {style}. Choose from 'satellite', 'white', or 'regular'.")

    selected_style = tile_styles[style]

    m = folium.Map(
        location=[40.75, -73.97],
        zoom_start=12,
        tiles=selected_style['tiles'],
        attr=selected_style['attr']
    )


    for _, row in aggregated_data.iterrows():
        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
            popup_html = f"""
            <div style="font-family: 'Helvetica Neue', sans-serif; font-size: 14px;">
                <b style="color: #2c3e50;">{row['Detection Group']}</b><br>
                <span style="color: #16a085;">Total CRZ Entries:</span> <b>{row['CRZ Entries']}</b>
            </div>
            """
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="blue", icon="car", prefix="fa")
            ).add_to(m)

    # Add layer control for switching base maps
    # folium.LayerControl(position='topright', collapsed=False).add_to(m)

    return m



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
    
    # Map the detailed vehicle classes to contracted names
    vehicle_class_map = {
        "TLC Taxi/FHV": "TLC Taxi/FHV",
        "5 - Motorcycles": "Motorcycles",
        "4 - Buses": "Buses",
        "3 - Multi-Unit Trucks": "Trucks (Multi)",
        "2 - Single-Unit Trucks": "Trucks (Single)",
        "1 - Cars, Pickups and Vans": "Cars, Pickups, Vans"
    }
    grouped["Vehicle Class"] = grouped["Vehicle Class"].map(vehicle_class_map)
    
    # Pivot the data for proportions
    pivot = grouped.pivot(index="Vehicle Class", columns="Time Period", values="CRZ Entries").fillna(0)
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)  # Normalize to proportions

    # Reset index for Seaborn compatibility
    pivot_prop = pivot_prop.reset_index()

    # Melt the data for Seaborn's barplot
    melted_data = pivot_prop.melt(id_vars="Vehicle Class", var_name="Time Period", value_name="Proportion")

    # Plot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=melted_data,
        x="Vehicle Class",
        y="Proportion",
        hue="Time Period",
        palette={"Peak": "#FF5733", "Overnight": "#3498DB"},
        edgecolor="black"
    )
    
    # Customize the plot
    plt.title("Proportion of CRZ Entries by Time Period for Each Vehicle Class", fontsize=16, pad=20)
    plt.xlabel("Vehicle Class", fontsize=14, labelpad=10)
    plt.ylabel("Proportion of Total CRZ Entries", fontsize=14, labelpad=10)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Time Period", fontsize=12, title_fontsize=14, loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Display the plot in Streamlit
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
        # Ensure 'Toll Date' is datetime
    df['Toll Date'] = pd.to_datetime(df['Toll Date'], format='%m/%d/%Y')

    # Create enhanced features (traffic, calendar, weather) and run clustering (as before)
    daily_volume = df.groupby('Toll Date')['CRZ Entries'].sum().rename('total_volume')
    hourly_pivot = df.groupby(['Toll Date', 'Hour of Day'])['CRZ Entries'].sum().unstack(fill_value=0)
    
    peak_hours = [7, 8, 9, 17, 18, 19]
    peak_volume = hourly_pivot[peak_hours].sum(axis=1)
    peak_ratio = peak_volume / daily_volume
    hourly_std = hourly_pivot.std(axis=1)
    growth_rate = daily_volume.pct_change().fillna(0)
    is_weekend = (daily_volume.index.weekday >= 5).astype(int)

    API_KEY = "b1b230ed8664410080803403250504"
    BASE_URL = "http://api.weatherapi.com/v1/history.json"

    
    
    def fetch_weather(date_str, api_key=API_KEY, location="New York"):
        params = {"key": api_key, "q": location, "dt": date_str}
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        forecast_day = data["forecast"]["forecastday"][0]["day"]
        return {
            "date": date_str,
            "avgtemp_c": forecast_day["avgtemp_c"],
            "avghumidity": forecast_day["avghumidity"],
            "totalprecip_mm": forecast_day["totalprecip_mm"]
        }
    
    unique_dates = daily_volume.index.strftime('%Y-%m-%d').unique()
    weather_records = []
    for date in unique_dates:
        try:
            weather_data = fetch_weather(date)
            weather_records.append(weather_data)
        except Exception as e:
            print(f"Failed to fetch weather for {date}: {e}")
    
    weather_df = pd.DataFrame(weather_records)
    weather_df['Toll Date'] = pd.to_datetime(weather_df['date'], format='%Y-%m-%d')
    weather_df = weather_df.set_index('Toll Date')[['avgtemp_c','avghumidity','totalprecip_mm']]
    weather_df['discomfort_index'] = weather_df['avgtemp_c'] - ((100 - weather_df['avghumidity'])/5)
    
    features_df = pd.DataFrame({
        'total_volume': daily_volume,
        'peak_ratio': peak_ratio,
        'hourly_std': hourly_std,
        'growth_rate': growth_rate,
        'is_weekend': is_weekend
    })
    features_df = features_df.merge(weather_df, left_index=True, right_index=True, how='inner')
    
    print("\nEnhanced Feature Set (first 5 rows):")
    print(features_df.head())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Number of PCA components: {X_pca.shape[1]}")
    
    sil_scores = {}
    for k in range(2, 10):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_temp.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        sil_scores[k] = score
        print(f"Silhouette score for k={k}: {score:.4f}")
    optimal_k = max(sil_scores, key=sil_scores.get)
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    features_df['Cluster'] = cluster_labels
    
    cluster_summary = features_df.groupby('Cluster').mean()
    print("\nCluster Summary:")
    print(cluster_summary)
    
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap='viridis', s=50)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of Enhanced Clusters")
    plt.colorbar(label="Cluster")
    plt.show()
    
    descriptive_labels = {}
    for cluster in cluster_summary.index:
        if cluster_summary.loc[cluster, 'is_weekend'] >= 0.5:
            descriptive_labels[cluster] = "Weekend/Low Demand"
        else:
            descriptive_labels[cluster] = "Weekday"
    features_df['Pattern Type'] = features_df['Cluster'].map(descriptive_labels)
    
    print("\nAssigned Cluster Descriptive Labels (first 5 rows):")
    print(features_df[['Cluster', 'Pattern Type']].head())
    
    # ---------------------------
    # STEP C: Merge Cluster Labels with Hourly Traffic Data for Actionable Visualization
    # ---------------------------
    # Define daily_pattern: a pivot table of hourly CRZ Entries by Toll Date.
    daily_pattern = df.groupby(['Toll Date', 'Hour of Day'])['CRZ Entries'].sum().unstack(fill_value=0)
    print("\nDaily Traffic Pattern (first 5 rows):")
    print(daily_pattern.head())
    
    # Reset index to merge with features_df (which contains cluster labels)
    daily_pattern_reset = daily_pattern.reset_index()  
    features_reset = features_df.reset_index()  # features_df's index is Toll Date
    
    # Merge the cluster labels into daily_pattern based on Toll Date.
    merged_pattern = pd.merge(daily_pattern_reset, features_reset[['Toll Date', 'Cluster']], on='Toll Date', how='left')
    merged_pattern.set_index('Toll Date', inplace=True)
    
    # Group by cluster and calculate the average hourly traffic pattern.
    hour_cols = [col for col in merged_pattern.columns if isinstance(col, (int, float))]
    cluster_profiles_line = merged_pattern.groupby('Cluster')[hour_cols].mean()
    
    # Plot the average daily traffic profile for each cluster.
    plt.figure(figsize=(10,6))
    for c_label in cluster_profiles_line.index:
        plt.plot(hour_cols, cluster_profiles_line.loc[c_label, hour_cols],
                 label=f"{descriptive_labels.get(c_label, 'Unknown')}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average CRZ Entries")
    plt.title("Average Daily Traffic Pattern by Enhanced Clusters")
    plt.legend()
    plt.show()
    
    print("\nEnhanced Temporal Signature Identification complete.")
    print("These clusters incorporate traffic, weather, and calendar features to distinguish different types of days, visualized as hourly traffic profiles.")
