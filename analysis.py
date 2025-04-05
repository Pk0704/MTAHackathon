import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
import datetime

df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")

'''
print("Data sample:")
print(df.head(10))
'''
# ---------------------------
# Define Toll Rates for Different Vehicle Types
# ---------------------------
# Provided toll rates:
# - Passenger and small commercial vehicles (Sedans, SUVs, pick-up trucks, and small vans): 
#     Peak: $9, Overnight: $2.25
# - Motorcycles: 
#     Peak: $4.50, Overnight: $1.05
# - Trucks and buses: 
#     Peak: $14.40 - $21.60 (using average $18.0), Overnight: $3.60 - $5.40 (using average $4.5)
toll_rates = {
    'Passenger': {'Peak': 9.0, 'Overnight': 2.25},
    'Motorcycle': {'Peak': 4.50, 'Overnight': 1.05},
    'Truck': {'Peak': 18.0, 'Overnight': 4.5}
}

# ---------------------------
# Define Vehicle Groups Based on "Vehicle Class"
# ---------------------------
# We now separate into six groups with contracted names:
# "TLC Taxi/FHV", "5 - Motorcycles" (→ motorcycles),
# "4 - Buses" (→ buses),
# "3 - Multi-Unit Trucks" (→ Trucks (multi)),
# "2 - Single-Unit Trucks" (→ trucks (Single)),
# "1 - Cars, Pickups and Vans" (→ cars, pickups, vans)
taxis = df[df['Vehicle Class'] == 'TLC Taxi/FHV']
motorcycles = df[df['Vehicle Class'] == '5 - Motorcycles']
buses = df[df['Vehicle Class'] == '4 - Buses']
trucks_multi = df[df['Vehicle Class'] == '3 - Multi-Unit Trucks']
trucks_single = df[df['Vehicle Class'] == '2 - Single-Unit Trucks']
cars = df[df['Vehicle Class'] == '1 - Cars, Pickups and Vans']

print("\nTotal CRZ Entries (weighted counts) by Vehicle Group:")
print("TLC Taxi/FHV:", taxis['CRZ Entries'].sum())
print("Motorcycles:", motorcycles['CRZ Entries'].sum())
print("Buses:", buses['CRZ Entries'].sum())
print("Trucks (multi):", trucks_multi['CRZ Entries'].sum())
print("Trucks (Single):", trucks_single['CRZ Entries'].sum())
print("Cars, Pickups, Vans:", cars['CRZ Entries'].sum())

# ---------------------------
# PART 1: Weighted Analysis of Entry Times (Hour of Day)
# ---------------------------
def weighted_avg_hour(data):
    total_entries = data['CRZ Entries'].sum()
    if total_entries == 0:
        return None  # Avoid division by zero
    return (data['Hour of Day'] * data['CRZ Entries']).sum() / total_entries

avg_hour_taxis = weighted_avg_hour(taxis)
avg_hour_motorcycles = weighted_avg_hour(motorcycles)
avg_hour_buses = weighted_avg_hour(buses)
avg_hour_trucks_multi = weighted_avg_hour(trucks_multi)
avg_hour_trucks_single = weighted_avg_hour(trucks_single)
avg_hour_cars = weighted_avg_hour(cars)

print("\nWeighted Average Entry Hour:")
print("TLC Taxi/FHV:", avg_hour_taxis)
print("Motorcycles:", avg_hour_motorcycles)
print("Buses:", avg_hour_buses)
print("Trucks (multi):", avg_hour_trucks_multi)
print("Trucks (Single):", avg_hour_trucks_single)
print("Cars, Pickups, Vans:", avg_hour_cars)

plt.figure(figsize=(10, 6))
bins = range(0, 25)  # Hours 0 to 24

plt.hist(taxis['Hour of Day'], bins=bins, weights=taxis['CRZ Entries'], 
         alpha=0.5, label='TLC Taxi/FHV', edgecolor='black')
plt.hist(motorcycles['Hour of Day'], bins=bins, weights=motorcycles['CRZ Entries'], 
         alpha=0.5, label='Motorcycles', edgecolor='black')
plt.hist(buses['Hour of Day'], bins=bins, weights=buses['CRZ Entries'], 
         alpha=0.5, label='Buses', edgecolor='black')
plt.hist(trucks_multi['Hour of Day'], bins=bins, weights=trucks_multi['CRZ Entries'], 
         alpha=0.5, label='Trucks (multi)', edgecolor='black')
plt.hist(trucks_single['Hour of Day'], bins=bins, weights=trucks_single['CRZ Entries'], 
         alpha=0.5, label='Trucks (Single)', edgecolor='black')
plt.hist(cars['Hour of Day'], bins=bins, weights=cars['CRZ Entries'], 
         alpha=0.5, label='Cars, Pickups, Vans', edgecolor='black')

plt.xlabel("Hour of Day")
plt.ylabel("Number of Entries (Weighted by CRZ Entries)")
plt.title("Weighted Vehicle Entries by Hour of Day")
plt.legend()
plt.xticks(bins)
plt.show()

# ---------------------------
# PART Y: Vehicle Time Period Proportion Analysis
# ---------------------------
# Group CRZ Entries by Time Period and Vehicle Class and then contract the names.
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
print("\nVehicle Time Period Proportions (Contracted Names):")
print(pivot_prop)

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
plt.show()

# ---------------------------
# PART 2: Time Period Distribution Analysis
# ---------------------------
def time_period_distribution(data, group_name):
    dist = data.groupby('Time Period')['CRZ Entries'].sum()
    total = dist.sum()
    print(f"\nTime Period Distribution for {group_name}:")
    print(dist)
    print("Proportions:")
    print(dist / total)
    return dist

dist_taxis = time_period_distribution(taxis, "TLC Taxi/FHV")
dist_motorcycles = time_period_distribution(motorcycles, "motorcycles")
dist_buses = time_period_distribution(buses, "buses")
dist_trucks_multi = time_period_distribution(trucks_multi, "Trucks (multi)")
dist_trucks_single = time_period_distribution(trucks_single, "trucks (Single)")
dist_cars = time_period_distribution(cars, "cars, pickups, vans")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
groups = [('TLC Taxi/FHV', dist_taxis), 
          ('motorcycles', dist_motorcycles), 
          ('buses', dist_buses), 
          ('Trucks (multi)', dist_trucks_multi), 
          ('trucks (Single)', dist_trucks_single), 
          ('cars, pickups, vans', dist_cars)]
for ax, (group_name, dist) in zip(axes, groups):
    dist.plot(kind='bar', ax=ax, title=f"{group_name} Time Period Distribution")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Total CRZ Entries")
plt.tight_layout()
plt.show()

'''
# ---------------------------
# PART 3: Congestion Relief Zone Efficiency Analysis (Revenue Related)
# ---------------------------
print("\n--- Congestion Relief Zone Efficiency Analysis ---")
...
'''

# ---------------------------
# PART X: Enhanced Temporal Signature Identification
# ---------------------------
print("\n--- Enhanced Temporal Signature Identification ---")

# Ensure 'Toll Date' is datetime
df['Toll Date'] = pd.to_datetime(df['Toll Date'], format='%m/%d/%Y')

# 1. Traffic Features:
# Total daily volume
daily_volume = df.groupby('Toll Date')['CRZ Entries'].sum().rename('total_volume')

# Hourly traffic pivot
hourly_pivot = df.groupby(['Toll Date', 'Hour of Day'])['CRZ Entries'].sum().unstack(fill_value=0)

# Define peak hours (e.g., morning: 7-9 and evening: 17-19)
peak_hours = [7, 8, 9, 17, 18, 19]
peak_volume = hourly_pivot[peak_hours].sum(axis=1)
peak_ratio = peak_volume / daily_volume

# Variance of hourly traffic per day
hourly_std = hourly_pivot.std(axis=1)

# Daily growth rate (percentage change from previous day)
growth_rate = daily_volume.pct_change().fillna(0)

# 2. Calendar Effects:
# Weekend indicator (Saturday=5, Sunday=6)
is_weekend = (daily_volume.index.weekday >= 5).astype(int)

# 3. Weather Integration (Optional):
# For demonstration, let's fetch weather for each day.
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
        "totalprecip_mm": forecast_day["totalprecip_mm"]
    }

unique_dates = daily_volume.index.strftime('%Y-%m-%d').unique()
weather_records = []
for date in unique_dates:
    try:
        weather_data = fetch_weather(date)
        weather_records.append(weather_data)
        print(f"Fetched weather for {date}")
    except Exception as e:
        print(f"Failed to fetch weather for {date}: {e}")

weather_df = pd.DataFrame(weather_records)
weather_df['Toll Date'] = pd.to_datetime(weather_df['date'], format='%Y-%m-%d')
weather_df = weather_df.set_index('Toll Date')[['avgtemp_c','avghumidity','totalprecip_mm']]
# Compute a simple discomfort index: DI = avgtemp_c - ((100 - avghumidity)/5)
weather_df['discomfort_index'] = weather_df['avgtemp_c'] - ((100 - weather_df['avghumidity'])/5)

# 4. Combine all features into one DataFrame.
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

# 5. Scale the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

# 6. Dimensionality Reduction using PCA (retain 95% variance)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Number of PCA components: {X_pca.shape[1]}")

# 7. Determine optimal number of clusters using silhouette analysis.
sil_scores = {}
from sklearn.metrics import silhouette_score
for k in range(2, 10):
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels = kmeans_temp.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores[k] = score
    print(f"Silhouette score for k={k}: {score:.4f}")
optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\nOptimal number of clusters: {optimal_k}")

# 8. Run KMeans clustering.
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)
features_df['Cluster'] = cluster_labels

# 9. Compute cluster profiles (means) for interpretation.
cluster_summary = features_df.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

# 10. Visualize clusters using the first two principal components.
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap='viridis', s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Enhanced Clusters")
plt.colorbar(label="Cluster")
plt.show()

# 11. (Optional) Assign descriptive names to clusters.
descriptive_labels = {}
for cluster in cluster_summary.index:
    if cluster_summary.loc[cluster, 'peak_ratio'] > 0.6 and cluster_summary.loc[cluster, 'is_weekend'] < 0.5:
        descriptive_labels[cluster] = "High Demand Weekday"
    elif cluster_summary.loc[cluster, 'is_weekend'] > 0.5:
        descriptive_labels[cluster] = "Weekend/Low Demand"
    else:
        descriptive_labels[cluster] = "Mixed Pattern"
features_df['Pattern Type'] = features_df['Cluster'].map(descriptive_labels)
print("\nAssigned Cluster Descriptive Labels (first 5 rows):")
print(features_df[['Cluster', 'Pattern Type']].head())

print("\nEnhanced Temporal Signature Identification complete.")
print("These clusters incorporate traffic, weather, and calendar features to distinguish different types of days.")

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
print("\nMerged Data sample:")
print(merged_df.head())

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
print(f"\nR-squared: {r_squared:.4f}")

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

# Graph 2: Traffic Volume vs. Average Temperature (Color-coded by Bundled Weather Category)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', hue='weather_category', 
                palette='Set1', s=100, edgecolor='black')
sns.regplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', scatter=False, 
            color='black', line_kws={'linewidth':1.5})
plt.title('Daily Traffic Volume vs. Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Daily CRZ Entries')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Weather Category')
plt.tight_layout()
plt.show()
