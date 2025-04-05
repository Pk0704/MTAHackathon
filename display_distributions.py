import pandas as pd
import streamlit as st
from perspective import Table
import streamlit.components.v1 as components

data=pd.read_csv('data.csv')

def display_vehicles(data, location):
    # Filter data by location
    df = data[data['Detection Group'] == location]
    
    # Create a Perspective table using the factory function
    table = Table(df)

    # Generate HTML/JS for the Perspective viewer
    html_template = """
    <link href='https://unpkg.com/@finos/perspective-viewer/dist/css/material.css' rel='stylesheet'>
    <script src='https://unpkg.com/@finos/perspective-viewer/dist/umd/perspective-viewer.js'></script>
    <perspective-viewer style="height: 500px; width: 100%;" id="viewer"></perspective-viewer>
    <script>
      const viewer = document.getElementById("viewer");
      const table = perspective.worker().table(%s);
      viewer.load(table);
      viewer.restore({group_by: ["Vehicle Class"], columns: ["Count"]});
    </script>
    """ % table.to_arrow().to_pybytes().hex()

    # Render in Streamlit
    components.html(html_template, height=600)
    
def display_time_series(data, location):
    
    df = data[data['Detection Group'] == location]
    
    # Display the filtered data
    st.write("Distributions: ")
    
    # Get the distribution of vehicle classes
    vehicle_counts = df['Toll Hour'].value_counts()
    
    # Display the bar chart
    st.bar_chart(vehicle_counts)
    
display_vehicles(data, 'Brooklyn Bridge')