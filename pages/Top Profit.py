from models.functions.top_profit import get_total_revenue

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Predicted Profit For All Flowers")

# Date input widgets
start_date = st.date_input("Select Start Date")
end_date = st.date_input("Select End Date")

# Ensure valid date range
if start_date and end_date:
    if start_date > end_date:
        st.error("Start date must be earlier than or equal to the end date.")
    else:
        # Fetch total profit
        total_profit_dict = get_total_revenue(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        # Convert dictionary to DataFrame
        profit_df = pd.DataFrame(list(total_profit_dict.items()), columns=['Flower Name', 'Predicted Profit'])
        st.subheader("Predicted Profit Table")
        st.dataframe(profit_df)

        # Bar chart using matplotlib
        st.subheader("Predicted Profit (Bar Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(profit_df['Flower Name'], profit_df['Predicted Profit'], color='lightblue', alpha=0.8)
        ax.set_title("Predicted Profit by Flower", fontsize=16)
        ax.set_xlabel("Flower Name", fontsize=12)
        ax.set_ylabel("Predicted Profit", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

        # Pie chart using matplotlib
        st.subheader("Profit Distribution (Pie Chart)")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(profit_df['Predicted Profit'], labels=profit_df['Flower Name'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title("Profit Distribution by Flower", fontsize=16)
        st.pyplot(fig)
else:
    st.warning("Please select both start and end dates.")