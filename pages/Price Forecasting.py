from models.functions.price_forecasting import predict_for_days
from models.functions.price_forecasting import generate_forecast_summary
from models.functions.price_forecasting import find_best_flower

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Flower Revenue and Profit Forecast")

# User input for the number of days
days = st.number_input("Enter the number of days for prediction", min_value=1, max_value=365, value=30)

# Display the results
if days:
    st.subheader(f"Forecast for the next {days} days")
    
    # Generate forecast summary
    forecast_summary = generate_forecast_summary(days)
    st.dataframe(forecast_summary)

    # Find the best flower based on revenue and profit
    best_flower_revenue, max_revenue, best_flower_profit, max_profit = find_best_flower(days)

    # Display the best flower results
    st.markdown(f"**Flower with the highest predicted revenue:** {best_flower_revenue} (₹{max_revenue:.2f})")
    st.markdown(f"**Flower with the highest predicted profit:** {best_flower_profit} (₹{max_profit:.2f})")

    # Plot revenue distribution (Pie Chart)
    st.subheader("Revenue Distribution (Pie Chart)")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(forecast_summary['Total Predicted Revenue'], labels=forecast_summary['Flower Name'], 
           autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title("Revenue Distribution by Flower", fontsize=16)
    st.pyplot(fig)

    # Plot profit distribution (Bar Chart)
    st.subheader("Profit Distribution (Bar Chart)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(forecast_summary['Flower Name'], forecast_summary['Total Predicted Profit'], color='lightblue', alpha=0.8)
    ax.set_title("Profit Distribution by Flower", fontsize=16)
    ax.set_xlabel("Flower Name", fontsize=12)
    ax.set_ylabel("Predicted Profit", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)