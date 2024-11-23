import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = 'data/flowers_dataset_cleaned.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Convert datetime columns
data['Start DateTime'] = pd.to_datetime(data['Start DateTime'])
data['End DateTime'] = pd.to_datetime(data['End DateTime'])

# Streamlit app
st.title("Dynamic Flower MRP Visualization with Plotly")

# Dropdown menu for selecting flower
flowers = data['Flower Name'].unique()
selected_flower = st.selectbox("Select a flower:", flowers)

# Date range selector
start_date = st.date_input("Start date:", value=data['Start DateTime'].min().date())
end_date = st.date_input("End date:", value=data['Start DateTime'].max().date())

# Filter data based on user inputs
filtered_data = data[
    (data['Flower Name'] == selected_flower) &
    (data['Start DateTime'].dt.date >= start_date) &
    (data['Start DateTime'].dt.date <= end_date)
]

# Update plot dynamically
st.subheader(f"MRP of {selected_flower} from {start_date} to {end_date}")

if filtered_data.empty:
    st.warning("No data available for the selected flower and date range.")
else:
    # Sort data by Start DateTime for a proper line graph
    filtered_data = filtered_data.sort_values(by='Start DateTime')

    # Plotting the line graph with Plotly
    fig = px.line(
        filtered_data,
        x='Start DateTime',
        y='MRP (₹)',
        title=f"MRP Trend for {selected_flower}",
        labels={'Start DateTime': 'Date', 'MRP (₹)': 'MRP (₹)'},
        markers=True
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="MRP (₹)",
        title_x=0.5,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
