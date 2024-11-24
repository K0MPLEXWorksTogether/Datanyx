import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the joblib model
def load_model(filename='models/regression/adjusted_flower_sales_model.joblib'):
    """Load the saved flower sales optimization model."""
    return joblib.load(filename)

# Profit calculation function
def calculate_profit(price, cost, quantity):
    """Calculate profit for a given price, cost, and quantity."""
    return (price - cost) * quantity

# Load flower dataset
def load_flower_data(dataset_path="data/flowers_dataset_cleaned.csv"):
    """Load the flower dataset."""
    return pd.read_csv(dataset_path)

# Filter data by the provided start and end dates
def filter_data_by_dates(data, start_date, end_date):
    """Filter the flower data to the given date range."""
    data['Start DateTime'] = pd.to_datetime(data['Start DateTime'])
    data['End DateTime'] = pd.to_datetime(data['End DateTime'])
    
    filtered_data = data[(data['Start DateTime'] >= start_date) & (data['End DateTime'] <= end_date)]
    return filtered_data

# Function to get optimal sales based on the provided start_date and end_date
def get_optimal_sales(start_date, end_date, model, dataset_path="data/flowers_dataset_cleaned.csv"):
    """Get the optimal flower sales for the given date range."""
    
    # Load flower data
    data = load_flower_data(dataset_path)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter the dataset by the date range
    filtered_data = filter_data_by_dates(data, start_date, end_date)
    
    if filtered_data.empty:
        st.write("No data found for the given date range.")
        return {}

    flower_sales = []
    cost_per_unit = 50  # Example cost per unit

    # For each flower, calculate the total profit based on the optimal quantities from the model
    for flower_name in filtered_data['Flower Name'].unique():
        flower_data = filtered_data[filtered_data['Flower Name'] == flower_name]
        
        best_quantity = model.get(flower_name, 0)
        total_profit = 0
        
        for _, row in flower_data.iterrows():
            daily_price = row['MRP (₹)']
            total_profit += calculate_profit(daily_price, cost_per_unit, best_quantity)
        
        flower_sales.append({
            'Flower Name': flower_name,
            'Best Quantity': best_quantity,
            'Total Profit (₹)': total_profit
        })

    return flower_sales

# Streamlit interface
st.title('Flower Sales Optimization')

# Load model
model = load_model('models/regression/adjusted_flower_sales_model.joblib')

# User inputs for date range
start_date = st.date_input('Select start date', pd.to_datetime('2023-11-01'))
end_date = st.date_input('Select end date', pd.to_datetime('2023-11-30'))

# Button to get optimized sales
if st.button('Get Optimized Sales'):
    # Get the optimized sales for the selected date range
    result = get_optimal_sales(start_date, end_date, model)
    
    # Display the result
    if result:
        # Display data in a table format
        df = pd.DataFrame(result)
        st.write("Optimized flower sales for the given date range:")
        st.dataframe(df)  # Display as a table

        # Bar Chart: Best Quantity for Each Flower using Matplotlib
        st.subheader('Best Quantity for Each Flower (Bar Chart)')
        fig, ax = plt.subplots()
        ax.bar(df['Flower Name'], df['Best Quantity'], color='skyblue')
        ax.set_xlabel('Flower Name')
        ax.set_ylabel('Best Quantity')
        ax.set_title('Best Quantity for Each Flower')
        plt.xticks(rotation=45, ha="right")  # Rotate flower names for better readability
        st.pyplot(fig)

        # Pie Chart: Distribution of Total Profit for Each Flower using Matplotlib
        st.subheader('Total Profit Distribution (Pie Chart)')
        fig, ax = plt.subplots()
        ax.pie(df['Total Profit (₹)'], labels=df['Flower Name'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    else:
        st.write("No results to display.")
