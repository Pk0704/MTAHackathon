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
cars = df[df['Vehicle Class'] == '1 - Cars, Pickups and Vans']
trucks = df[df['Vehicle Class'].isin(['2 - Single-Unit Trucks', '3 - Multi-Unit Trucks'])]
motorcycles = df[df['Vehicle Class'] == '5 - Motorcycles']
taxis = df[df['Vehicle Class'] == 'TLC Taxi/FHV']

print("\nTotal CRZ Entries (weighted counts) by Vehicle Group:")
print("Cars:", cars['CRZ Entries'].sum())
print("Trucks (Lorries):", trucks['CRZ Entries'].sum())
print("Motorcycles:", motorcycles['CRZ Entries'].sum())
print("Taxis:", taxis['CRZ Entries'].sum())

# ---------------------------
# PART 1: Weighted Analysis of Entry Times (Hour of Day)
# ---------------------------
def weighted_avg_hour(data):
    total_entries = data['CRZ Entries'].sum()
    if total_entries == 0:
        return None  # Avoid division by zero
    return (data['Hour of Day'] * data['CRZ Entries']).sum() / total_entries

avg_hour_cars = weighted_avg_hour(cars)
avg_hour_trucks = weighted_avg_hour(trucks)
avg_hour_motorcycles = weighted_avg_hour(motorcycles)
avg_hour_taxis = weighted_avg_hour(taxis)

print("\nWeighted Average Entry Hour:")
print("Cars:", avg_hour_cars)
print("Trucks (Lorries):", avg_hour_trucks)
print("Motorcycles:", avg_hour_motorcycles)
print("Taxis:", avg_hour_taxis)

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

dist_cars = time_period_distribution(cars, "Cars")
dist_trucks = time_period_distribution(trucks, "Trucks (Lorries)")
dist_motorcycles = time_period_distribution(motorcycles, "Motorcycles")
dist_taxis = time_period_distribution(taxis, "Taxis")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
groups = [('Cars', dist_cars), 
          ('Trucks (Lorries)', dist_trucks), 
          ('Motorcycles', dist_motorcycles), 
          ('Taxis', dist_taxis)]

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

# Define hypothetical capacities (throughput capacity per time block) for each detection group.
capacity_dict = {
    'Brooklyn Bridge': 150,
    'West Side Highway at 60th St': 120,
    'West 60th St': 100,
    'Queensboro Bridge': 130,
    'Queens Midtown Tunnel': 110,
    'Lincoln Tunnel': 140,
    'Holland Tunnel': 130,
    'FDR Drive at 60th St': 100,
    'East 60th St': 90,
    'Williamsburg Bridge': 110,
    'Manhattan Bridge': 120,
    'Hugh L. Carey Tunnel': 100
}

default_peak_toll = toll_rates['Passenger']['Peak']

df_peak = df[df['Time Period'].str.lower() == 'peak']

if df_peak.empty:
    print("No peak time data available. Ensure that the 'Time Period' column is correctly labeled for peak periods.")
else:
    efficiency_records = []
    groups_peak = df_peak.groupby('Detection Group')
    for group_name, group_data in groups_peak:
        if group_name in capacity_dict:
            num_time_blocks = group_data.shape[0]
            total_excluded = group_data['Excluded Roadway Entries'].sum()
            total_capacity = capacity_dict[group_name] * num_time_blocks
            utilization_ratio = total_excluded / total_capacity if total_capacity > 0 else None
            underutilization = total_capacity - total_excluded if total_excluded < total_capacity else 0
            opportunity_cost = underutilization * default_peak_toll
            efficiency_records.append({
                'Detection Group': group_name,
                'Num Time Blocks': num_time_blocks,
                'Total Excluded Entries': total_excluded,
                'Total Capacity': total_capacity,
                'Utilization Ratio': utilization_ratio,
                'Underutilization': underutilization,
                'Opportunity Cost': opportunity_cost
            })
    efficiency_df = pd.DataFrame(efficiency_records)
    print("\nEfficiency Summary (Peak Time Only):")
    print(efficiency_df)
    
    underutilization_vector = efficiency_df['Underutilization'].tolist()
    l2_norm = math.sqrt(sum(x**2 for x in underutilization_vector))
    print("\nAggregate Inefficiency (L2 Norm of Underutilization):", l2_norm)
    
    alternatives = {
        'Brooklyn Bridge': 'Manhattan Bridge',
        'Manhattan Bridge': 'Brooklyn Bridge',
        'Queensboro Bridge': 'Lincoln Tunnel',
        'Queens Midtown Tunnel': 'Lincoln Tunnel',
        'Lincoln Tunnel': 'Holland Tunnel',
        'Holland Tunnel': 'Lincoln Tunnel'
    }
    
    print("\nAlternative Entry Suggestions (for Congested Points):")
    congestion_threshold = 1.0
    for idx, row in efficiency_df.iterrows():
        if row['Utilization Ratio'] >= congestion_threshold:
            current_group = row['Detection Group']
            alt = alternatives.get(current_group, None)
            if alt:
                print(f"- {current_group} is congested (Utilization Ratio: {row['Utilization Ratio']:.2f}).")
                print(f"  Consider entering via {alt} instead.")
            else:
                print(f"- {current_group} is congested (Utilization Ratio: {row['Utilization Ratio']:.2f}).")
                print("  No alternative entry suggestion available.")
                
    # ---------------------------
    # PART 4: Congestion Equity Index Analysis (Revenue Related)
    # ---------------------------
    print("\n--- Congestion Equity Index Analysis ---")
    group_to_region = df_peak.groupby('Detection Group')['Detection Region'].first().to_dict()
    efficiency_df['Detection Region'] = efficiency_df['Detection Group'].map(group_to_region)
    
    region_efficiency = efficiency_df.groupby('Detection Region').agg({
        'Total Capacity': 'sum',
        'Total Excluded Entries': 'sum',
        'Opportunity Cost': 'sum'
    }).reset_index()
    
    region_entries = df_peak.groupby('Detection Region')['CRZ Entries'].sum().reset_index().rename(columns={'CRZ Entries': 'Total CRZ Entries'})
    
    region_summary = pd.merge(region_efficiency, region_entries, on='Detection Region', how='left')
    
    region_summary['Fairness Coefficient'] = region_summary['Opportunity Cost'] / region_summary['Total CRZ Entries']
    
    print("\nCongestion Equity Index (Fairness Coefficient by Region):")
    print(region_summary[['Detection Region', 'Total CRZ Entries', 'Opportunity Cost', 'Fairness Coefficient']])
    
    plt.figure(figsize=(10, 6))
    plt.bar(region_summary['Detection Region'], region_summary['Fairness Coefficient'], color='skyblue', edgecolor='black')
    plt.xlabel("Detection Region (Community)")
    plt.ylabel("Fairness Coefficient (Opportunity Cost per CRZ Entry)")
    plt.title("Congestion Equity Index by Detection Region")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("\nIn my view, the Congestion Equity Index is a particularly cool point of analysis, as it provides insight into how congestion pricing affects different communities and reveals potential inequities in the distribution of congestion costs.")
'''

# ---------------------------
# PART X: Temporal Signature Identification
# ---------------------------
print("\n--- Temporal Signature Identification ---")
# The goal here is to detect distinctive daily traffic patterns that could signal holidays, special events, or typical weekdays.
# We aggregate the data by Toll Date and Hour of Day to create a daily "signature" vector.

# Create a pivot table with Toll Date as the index and Hour of Day as columns, summing CRZ Entries.
daily_pattern = df.groupby(['Toll Date', 'Hour of Day'])['CRZ Entries'].sum().unstack(fill_value=0)
print("\nDaily Traffic Pattern (first 5 rows):")
print(daily_pattern.head())

# Normalize the daily patterns for clustering.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
daily_pattern_scaled = scaler.fit_transform(daily_pattern)

# Use silhouette analysis to find the optimal number of clusters.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sil_scores = {}
for k in range(2, 10):
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    temp_clusters = kmeans_temp.fit_predict(daily_pattern_scaled)
    score = silhouette_score(daily_pattern_scaled, temp_clusters)
    sil_scores[k] = score
    print(f"Silhouette score for k={k}: {score:.4f}")

optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\nOptimal number of clusters determined by silhouette analysis: {optimal_k}")

# Now run K-Means with the optimal number of clusters.
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(daily_pattern_scaled)
daily_pattern['Cluster'] = clusters

print("\nDaily pattern clusters assigned (first 5 rows):")
print(daily_pattern[['Cluster']].head())

# Compute average daily signature for each cluster.
cluster_profiles = daily_pattern.groupby('Cluster').mean()

# --- Assign descriptive names to clusters based on a simple heuristic ---
# (This heuristic is arbitrary and can be refined.)
cluster_labels = {}
for cluster in cluster_profiles.index:
    # Compute the mean CRZ Entries during typical morning hours (e.g., 7-10)
    morning_hours = [hour for hour in cluster_profiles.columns if isinstance(hour, (int, float)) and 7 <= hour <= 10]
    morning_peak = cluster_profiles.loc[cluster, morning_hours].mean() if morning_hours else 0
    overall_max = cluster_profiles.loc[cluster, [col for col in cluster_profiles.columns if isinstance(col, (int, float))]].max()
    ratio = morning_peak / overall_max if overall_max > 0 else 0
    
    # Simple rules to name clusters (these thresholds are illustrative):
    if ratio > 0.6:
        label = "Weekday Pattern"
    elif ratio < 0.3:
        label = "Weekend/Holiday Pattern"
    else:
        label = "Mixed Pattern"
    cluster_labels[cluster] = label

# Map the descriptive labels back to the daily_pattern DataFrame.
daily_pattern['Pattern Type'] = daily_pattern['Cluster'].map(cluster_labels)
print("\nAssigned Cluster Labels (first 5 rows):")
print(daily_pattern[['Cluster', 'Pattern Type']].head())

# Plot the average daily traffic pattern for each cluster with descriptive names.
plt.figure(figsize=(10,6))
# Extract the hour columns (numeric columns) from the pivot table.
hour_columns = [col for col in daily_pattern.columns if isinstance(col, (int, float))]
for cluster in cluster_profiles.index:
    plt.plot(hour_columns, cluster_profiles.loc[cluster, hour_columns],
             label=f"Cluster {cluster} ({cluster_labels[cluster]})")
plt.xlabel("Hour of Day")
plt.ylabel("Average CRZ Entries")
plt.title("Average Daily Traffic Pattern Signature by Cluster")
plt.legend()
plt.show()

print("\nTemporal Signature Identification complete.")
print("These clusters represent distinctive daily traffic patterns that may correspond to normal weekdays, weekends, holidays, or special events.")


# --------------------------------------
# Weather API & Visual Analysis
# --------------------------------------
API_KEY = "b1b230ed8664410080803403250504"
BASE_URL = "http://api.weatherapi.com/v1/history.json"

# Function to fetch weather data for a given date (format: YYYY-MM-DD)
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

# ---------------------------
# Load Traffic Data (your existing dataset)
# ---------------------------
df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
print("Traffic Data sample:")
print(df.head())

# ---------------------------
# Aggregate Traffic Data by Toll Date
# ---------------------------
traffic_daily = df.groupby("Toll Date")["CRZ Entries"].sum().reset_index()
traffic_daily.rename(columns={"CRZ Entries": "daily_crz_entries"}, inplace=True)
print("\nDaily Traffic Totals (first 5 rows):")
print(traffic_daily.head())

# ---------------------------
# Fetch Weather Data for Each Toll Date
# ---------------------------
unique_dates = traffic_daily["Toll Date"].unique()
weather_records = []
for date in unique_dates:
    try:
        # Convert the date to the proper format if needed (assumes "MM/DD/YYYY")
        # Here, we assume the dates are already in "YYYY-MM-DD" format.
        weather_data = fetch_weather(date)
        weather_records.append(weather_data)
        print(f"Fetched weather for {date}")
    except Exception as e:
        print(f"Failed to fetch weather for {date}: {e}")

weather_df = pd.DataFrame(weather_records)
print("\nWeather Data sample:")
print(weather_df.head())

# ---------------------------
# Merge Traffic and Weather Data
# ---------------------------
merged_df = pd.merge(traffic_daily, weather_df, left_on="Toll Date", right_on="date", how="inner")
print("\nMerged Data sample:")
print(merged_df.head())

# ---------------------------
# Build a Regression Model: Predict Daily Traffic (CRZ Entries) based on Weather
# ---------------------------
# Use weather features: average temperature, average humidity, total precipitation, and condition.
features = ["avgtemp_c", "avghumidity", "totalprecip_mm", "condition"]
X = merged_df[features]
y = merged_df["daily_crz_entries"]

# Create a preprocessor to one-hot encode the "condition" column.
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), ["condition"])
], remainder="passthrough")  # Pass remaining columns as is.

# Build a pipeline that applies the preprocessor and then fits a linear regression model.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

pipeline.fit(X, y)

# Print out the model coefficients.
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

# Convert Toll Date to datetime
merged_df['Toll Date'] = pd.to_datetime(merged_df['Toll Date'], format='%m/%d/%Y')

# Graph 1: Daily Traffic Volume Over Time with Weather Conditions
plt.figure(figsize=(14, 6))
sns.lineplot(data=merged_df, x='Toll Date', y='daily_crz_entries', marker='o', color='gray', label='Daily Traffic')
sns.scatterplot(data=merged_df, x='Toll Date', y='daily_crz_entries', hue='condition',
                palette='Set2', s=100, edgecolor='black')
plt.title('Daily Traffic Volume Over Time with Weather Conditions')
plt.xlabel('Date')
plt.ylabel('Daily CRZ Entries')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Weather Condition')
plt.tight_layout()
plt.show()

# Graph 2: Traffic Volume vs. Average Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', hue='condition', palette='Set1', s=100, edgecolor='black')
sns.regplot(data=merged_df, x='avgtemp_c', y='daily_crz_entries', scatter=False, color='black', line_kws={'linewidth':1.5})
plt.title('Daily Traffic Volume vs. Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Daily CRZ Entries')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Weather Condition')
plt.tight_layout()
plt.show()

