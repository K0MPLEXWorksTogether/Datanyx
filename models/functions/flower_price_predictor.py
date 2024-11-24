import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Load necessary components
def load_components():
    data = pd.read_csv("hackathon/flowers_dataset_cleaned.csv")
    label_encoder = joblib.load('hackathon/Datanyx/models/regression/label_encoder.joblib')
    scaler_prices = joblib.load('hackathon/Datanyx/models/regression/scaler_prices.joblib')  # For scaling profit
    return data, label_encoder, scaler_prices

# Calculate daily profit for each flower (for demonstration, use the last 30 days)
def calculate_daily_profit(data):
    """
    Calculate the daily profit for each flower based on the last 30 days of data.
    For simplicity, we'll assume profits vary slightly day by day.
    """
    # Calculate daily profit as the average of MRP for each flower in the last 30 days
    daily_profit_per_flower = data.groupby('Flower Name')['MRP (₹)'].rolling(30).mean().reset_index()
    
    # Use the last available daily profit as the baseline profit for the next days
    daily_profit_per_flower = daily_profit_per_flower.groupby('Flower Name').last()['MRP (₹)']
    
    return daily_profit_per_flower

# Forecast day-wise profit for a flower
def forecast_daywise_profit(flower_id, n_days, data, label_encoder, daily_profit_per_flower):
    """
    Forecast the day-wise profit for a flower based on its name or ID and the number of days.
    """
    flower_names = data['Flower Name'].unique()
    
    if isinstance(flower_id, int):  # Input is an index
        if flower_id < 0 or flower_id >= len(flower_names):
            raise ValueError(f"Invalid flower ID. Choose between 0 and {len(flower_names) - 1}.")
        flower_name = flower_names[flower_id]
    elif isinstance(flower_id, str):  # Input is a flower name
        if flower_id not in flower_names:
            raise ValueError(f"Flower name '{flower_id}' not found in dataset.")
        flower_name = flower_id
    else:
        raise ValueError("Invalid flower identifier. Must be an integer ID or a string name.")
    
    # Get the baseline daily profit for the flower
    daily_profit = daily_profit_per_flower[flower_name]
    
    # Generate day-wise profit (we simulate slight fluctuations day-to-day)
    daywise_profit = []
    for day in range(1, n_days + 1):
        fluctuation = np.random.normal(loc=0, scale=0.05)  # Simulate slight random fluctuation
        day_profit = daily_profit * (1 + fluctuation)
        daywise_profit.append(round(day_profit, 2))
    
    # Format day-wise profit output
    daywise_profit_output = "\n".join([f"Day {i+1} : ₹{profit}" for i, profit in enumerate(daywise_profit)])
    
    return {
        "Flower Name": flower_name,
        "Days": n_days,
        "Day-wise Profits (₹)": daywise_profit_output
    }

# Wrapper for external usage
def predict_daywise_flower_profit(flower_id, n_days):
    """
    Wrapper function to predict day-wise flower profit for external usage.
    """
    data, label_encoder, scaler_prices = load_components()
    daily_profit_per_flower = calculate_daily_profit(data)
    
    try:
        result = forecast_daywise_profit(
            flower_id=flower_id,
            n_days=n_days,
            data=data,
            label_encoder=label_encoder,
            daily_profit_per_flower=daily_profit_per_flower
        )
        return result
    except Exception as e:
        return {"Error": str(e)}

# Example 1: Using flower index
result = predict_daywise_flower_profit(flower_id=1, n_days=7)
print(result["Day-wise Profits (₹)"])

# Example 2: Using flower name
result = predict_daywise_flower_profit(flower_id="Rose", n_days=7)
print(result["Day-wise Profits (₹)"])
