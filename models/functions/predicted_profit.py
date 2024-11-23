import pandas as pd
import joblib
from datetime import datetime

# Load the pre-saved prediction summary
prediction_summary = joblib.load('hackathon/Datanyx/models/regression/prediction_summary.joblib')

# Function to get predicted profit as dictionary based on start_date and end_date
def get_predicted_profit(start_date: str, end_date: str):
    # Convert string dates to datetime
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Ensure that 'Date' column is in datetime format
    prediction_summary['Date'] = pd.to_datetime(prediction_summary['Date'])

    # Filter the data based on the date range
    filtered_data = prediction_summary[(prediction_summary['Date'] >= start_date) & (prediction_summary['Date'] <= end_date)]

    # Create a dictionary with 'Date' as the key and 'Predicted Profit' as the value
    profit_dict = {(row['Flower Name']):row["Predicted Profit"]
    for _, row in filtered_data.iterrows()}
    return profit_dict

# Example usage:
start_date = "2024-11-01"
end_date = "2024-11-30"
predicted_profit_dict = get_predicted_profit(start_date, end_date)

# Print the predicted profit dictionary
print(predicted_profit_dict)
