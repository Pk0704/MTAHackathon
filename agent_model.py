from openai import OpenAI
import pandas as pd

client = OpenAI(api_key='')

with open("data.csv", "r") as f:
    csv_data = f.read()
df = pd.read_csv("data.csv")

print("Here's a preview of your CSV:\n")
print(df.head(), "\n")
print("These are the columns in your CSV:\n")
print(", ".join(df.columns))

user_request = input(
    "\nWhat would you like to visualize? (You can specify columns or say 'just show me something cool')\n> "
)

prompt = f"""
You are a skilled Python data analyst using pandas and matplotlib or seaborn. You also use GeoPandas and Folium for making interactive maps.

Based on the user input and the data, generate insightful charts or maps using the most relevant numerical and categorical columns to highlight trends, differences, or totals.
Avoid plots that don't convey meaning.

Here is a CSV file loaded as a pandas DataFrame, the dataset is called 'MTA Congestion Relief Zone Vehicle Entries: Beginning 2025.'
This dataset provides the number of crossings into the Congestion Relief Zone by crossing location and vehicle class, in 10-minute intervals.
Your job is to create a meaningful data visualization based on what the user wants. If they don't specify, create a compelling chart that best represents the trends in the dataset.
If you choose to create a map, use Folium and make sure to include coordinates for each location if possible. Use MarkerCluster to group map points, and save the map as an HTML file. Do not just return the map object — include code to save and open the map using webbrowser.

In the code you return, you should load the data like this: df = pd.read_csv("data.csv")
The visualization should use the entire dataset unless the user specifies otherwise.

User Request:
{user_request}

CSV Data (first rows):
{df.head().to_csv(index=False)}

Column Names:
{', '.join(df.columns)}

Only return valid Python code that uses pandas and matplotlib or seaborn to generate the visualization.
Do not include any explanations or markdown, just the code.
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Python data visualization assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

generated_code = response.choices[0].message.content
print("\nGenerated Python code:\n")
print(generated_code)