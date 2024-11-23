import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Load the cleaned dataset
data = pd.read_csv("hackathon/flowers_dataset_cleaned.csv")

# Check if the Timestamp column exists and convert it to datetime
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Encode flower names
label_encoder = LabelEncoder()
data['Flower Name Encoded'] = label_encoder.fit_transform(data['Flower Name'])
flower_names = label_encoder.classes_

# Load the trained revenue and profit models
revenue_model = joblib.load('hackathon/revenue_model_svm.joblib')
profit_model = joblib.load('hackathon/profit_model_svm.joblib')

# Function to get total predicted revenue within a specific date range
def get_total_revenue(start_date: str, end_date: str):
    """
    Calculate the total predicted revenue for each flower within a specific date range.

    Args:
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
    dict: A dictionary with flower names as keys and total predicted revenue as values.
    """
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate the date range between the start and end date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # List to hold the results
    results = []

    # Loop through each flower to generate predictions
    for flower in flower_names:
        encoded_flower = label_encoder.transform([flower])[0]  # Get encoded flower name
        freq_qty_sold = np.random.randint(50, 200, len(date_range))  # Random quantities sold
        average_price = data['MRP (₹)'].mean()  # Average price from the dataset

        # Prepare future data for prediction
        future_data = pd.DataFrame({
            'Flower Name': [encoded_flower] * len(date_range),
            'Qty Sold (kg)': freq_qty_sold,
            'MRP (₹)': [average_price] * len(date_range)
        })

        # Predict revenue using the revenue model
        predicted_revenue = revenue_model.predict(future_data)

        # Store the results for the current flower
        for i, single_date in enumerate(date_range):
            results.append({
                'Flower Name': flower,
                'Date': single_date,
                'Predicted Revenue': predicted_revenue[i]
            })

    # Create a DataFrame from the results
    prediction_summary = pd.DataFrame(results)

    # Aggregate the predicted revenue by flower name
    aggregated_revenue = prediction_summary.groupby('Flower Name')['Predicted Revenue'].sum().reset_index()

    # Sort the revenue in descending order to get top flowers by revenue
    top_revenue = aggregated_revenue.sort_values(by='Predicted Revenue', ascending=False)

    # Convert the result to a dictionary where Flower Name is the key and Predicted Revenue is the value
    revenue_dict = dict(zip(top_revenue['Flower Name'], top_revenue['Predicted Revenue']))

    return revenue_dict

start_date = "2024-11-01"
end_date = "2024-11-10"
revenue_dict = get_total_revenue(start_date, end_date)

print(revenue_dict)