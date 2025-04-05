import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data.csv")

# Convert 'Toll Hour' to datetime to extract hour
df['Toll Hour'] = pd.to_datetime(df['Toll Hour'])
df['Hour of Day'] = df['Toll Hour'].dt.hour

# Group by Hour of Day and Vehicle Class to find the dominant vehicle types
vehicle_hourly = df.groupby(['Hour of Day', 'Vehicle Class'])['CRZ Entries'].sum().reset_index()

# Find the dominant vehicle class for each hour
dominant_vehicles = vehicle_hourly.loc[vehicle_hourly.groupby('Hour of Day')['CRZ Entries'].idxmax()]

# Plot the dominant vehicle types by hour
plt.figure(figsize=(12, 6))
sns.barplot(data=dominant_vehicles, x='Hour of Day', y='CRZ Entries', hue='Vehicle Class')
plt.title('Dominant Vehicle Types by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('CRZ Entries')
plt.legend(title='Vehicle Class')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()