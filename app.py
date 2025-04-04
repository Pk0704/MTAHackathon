import streamlit as st
import pandas as pd

#title
st.title("CSV File Viewer")


# Load the CSV file
csv_file = "your_file_name.csv"  # Replace with your file's name
try:
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)
    
    # Display the DataFrame on the website
    st.write("Here's your CSV file displayed on the website:")
    st.dataframe(df)  # Interactive table
except FileNotFoundError:
    st.error(f"The file '{csv_file}' was not found. Please check the file path.")