import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Load necessary components
def load_components():
    data = pd.read_csv("data/flowers_dataset_cleaned.csv")
    label_encoder = joblib.load('models/regression/label_encoder.joblib')
    scaler_prices = joblib.load('models/regression/scaler_prices.joblib')  # For scaling profit
    return data, label_encoder, scaler_prices

# Calculate daily profit for each flower (for demonstration, use the last 30 days)
def calculate_daily_profit(data):
    daily_profit_per_flower = data.groupby('Flower Name')['MRP (₹)'].rolling(30).mean().reset_index()
    daily_profit_per_flower = daily_profit_per_flower.groupby('Flower Name').last()['MRP (₹)']
    return daily_profit_per_flower

# Forecast day-wise profit for a flower
def forecast_daywise_profit(flower_id, n_days, data, label_encoder, daily_profit_per_flower):
    flower_names = data['Flower Name'].unique()
    
    if isinstance(flower_id, int):
        if flower_id < 0 or flower_id >= len(flower_names):
            raise ValueError(f"Invalid flower ID. Choose between 0 and {len(flower_names) - 1}.")
        flower_name = flower_names[flower_id]
    elif isinstance(flower_id, str):
        if flower_id not in flower_names:
            raise ValueError(f"Flower name '{flower_id}' not found in dataset.")
        flower_name = flower_id
    else:
        raise ValueError("Invalid flower identifier. Must be an integer ID or a string name.")
    
    daily_profit = daily_profit_per_flower[flower_name]
    
    # Generate day-wise profit with fluctuations
    daywise_profit = []
    for day in range(1, n_days + 1):
        fluctuation = np.random.normal(loc=0, scale=0.05)  # Simulate slight random fluctuation
        day_profit = daily_profit * (1 + fluctuation)
        daywise_profit.append(round(day_profit, 2))
    
    # Return the formatted day-wise profit output
    daywise_profit_output = "\n".join([f"Day {i+1} : ₹{profit}" for i, profit in enumerate(daywise_profit)])
    
    return {
        "Flower Name": flower_name,
        "Days": n_days,
        "Day-wise Profits (₹)": daywise_profit_output,
        "Day-wise Profits (list)": daywise_profit
    }

# Streamlit app interface
def main():
    st.title("Flower Profit Forecast")
    st.write("Use this tool to forecast the daily profits of flowers based on their name or ID.")
    
    # Load components
    data, label_encoder, scaler_prices = load_components()
    daily_profit_per_flower = calculate_daily_profit(data)
    
    # Flower ID or Name input
    flower_input = st.text_input("Enter Flower Name or Flower ID:")
    n_days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=30, value=7)
    
    if flower_input:
        try:
            # Try if the input is an integer (ID)
            flower_id = int(flower_input)
            result = forecast_daywise_profit(flower_id, n_days, data, label_encoder, daily_profit_per_flower)
        except ValueError:
            # If it's not an integer, treat it as a flower name
            flower_id = flower_input
            result = forecast_daywise_profit(flower_id, n_days, data, label_encoder, daily_profit_per_flower)
        
        # Display the results
        if "Error" in result:
            st.error(result["Error"])
        else:
            # Table of results
            st.subheader(f"Day-wise profit forecast for {result['Flower Name']}")
            st.write(pd.DataFrame({
                "Day": [f"Day {i+1}" for i in range(n_days)],
                "Profit (₹)": result["Day-wise Profits (list)"]
            }))
            
            # Line Graph using Matplotlib
            fig, ax = plt.subplots()
            ax.plot(range(1, n_days + 1), result["Day-wise Profits (list)"], marker='o', color='b', label="Profit (₹)")
            ax.set_xlabel("Day")
            ax.set_ylabel("Profit (₹)")
            ax.set_title(f"Profit Forecast for {result['Flower Name']}")
            ax.legend()
            st.pyplot(fig)
            
            # Pie Chart for profit distribution across days
            profit_labels = [f"Day {i+1}" for i in range(n_days)]
            fig_pie = px.pie(values=result["Day-wise Profits (list)"], names=profit_labels, title=f"Profit Distribution for {result['Flower Name']}")
            st.plotly_chart(fig_pie)

if __name__ == "__main__":
    main()
