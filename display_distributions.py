from perspective import PerspectiveViewer
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd

data=pd.read_csv('data.csv')

def display_vehicles(data, location):
    # Filter data by location
    df = data[data['Detection Group'] == location]
    
    # Display the filtered data
    st.write("Distributions: ")
    
    # Create a Perspective table
    perspective_widget = PerspectiveViewer(df)
    perspective_widget.view = "y_bar"  # Set the view to a bar chart
    perspective_widget.columns = ["Vehicle Class", "Count"]  # Specify columns
    perspective_widget.row_pivots = ["Vehicle Class"]  # Group by vehicle class
    
    # Render the Perspective widget in Streamlit
    components.html(perspective_widget.to_html(), height=500)
    
def display_time_series(data, location):
    
    df = data[data['Detection Group'] == location]
    
    # Display the filtered data
    st.write("Distributions: ")
    
    # Get the distribution of vehicle classes
    vehicle_counts = df['Toll Hour'].value_counts()
    
    # Display the bar chart
    st.bar_chart(vehicle_counts)
    
display_vehicles(data, 'Brooklyn Bridge')