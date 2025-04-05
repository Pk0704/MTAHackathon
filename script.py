import pandas as pd

data=pd.read_csv('MTAHackathon\MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv')

new_data=data[:100]

new_data.to_csv('data.csv')