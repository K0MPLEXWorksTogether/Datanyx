import joblib
import pandas as pd

# Load the joblib model
def load_model(filename='hackathon/Datanyx/models/regression/adjusted_flower_sales_model.joblib'):
    """Load the saved flower sales optimization model."""
    return joblib.load(filename)

# Profit calculation function
def calculate_profit(price, cost, quantity):
    """Calculate profit for a given price, cost, and quantity."""
    return (price - cost) * quantity

# Load flower dataset
def load_flower_data(dataset_path="hackathon/flowers_dataset_cleaned.csv"):
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
def get_optimal_sales(start_date, end_date, model, dataset_path="hackathon/flowers_dataset_cleaned.csv"):
    """Get the optimal flower sales for the given date range."""
    
    # Load flower data
    data = load_flower_data(dataset_path)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter the dataset by the date range
    filtered_data = filter_data_by_dates(data, start_date, end_date)
    
    if filtered_data.empty:
        print("No data found for the given date range.")
        return {}

    flower_sales = {}
    cost_per_unit = 50  # Example cost per unit

    # For each flower, calculate the total profit based on the optimal quantities from the model
    for flower_name in filtered_data['Flower Name'].unique():
        flower_data = filtered_data[filtered_data['Flower Name'] == flower_name]
        
        best_quantity = model.get(flower_name, 0)
        total_profit = 0
        
        for _, row in flower_data.iterrows():
            daily_price = row['MRP (₹)']
            total_profit += calculate_profit(daily_price, cost_per_unit, best_quantity)
        
        flower_sales[flower_name] = best_quantity

    return flower_sales

# Example Usage
if __name__ == "__main__":
    model = load_model('hackathon/Datanyx/models/regression/adjusted_flower_sales_model.joblib')  # Load the pre-saved model
    start_date = '2023-11-01'  # Define the start date
    end_date = '2023-11-30'    # Define the end date
    
    # Get the optimized sales for the provided date range
    result = get_optimal_sales(start_date, end_date, model)
    
    # Print the result
    print(result)
