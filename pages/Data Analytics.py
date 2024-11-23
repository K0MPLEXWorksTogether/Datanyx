import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Importing the necessary functions from each file
from models.functions.predicted_profit import get_predicted_profit
from models.functions.predicted_revenue import get_aggregated_results
from models.functions.price_forecasting import predict_for_days, generate_forecast_summary, find_best_flower
from models.functions.top_profit import get_total_revenue

# Dropdown menu for feature selection
feature_option = st.sidebar.selectbox("Select the feature to display", 
                                     ["Flower MRP Visualization", "Predicted Profit", "Predicted Revenue", 
                                      "Flower Revenue and Profit Forecast", "Predicted Profit for All Flowers"])

# Flower MRP Visualization
if feature_option == "Flower MRP Visualization":
    file_path = 'data/flowers_dataset_cleaned.csv'  # Update with the actual path
    data = pd.read_csv(file_path)
    data['Start DateTime'] = pd.to_datetime(data['Start DateTime'])
    data['End DateTime'] = pd.to_datetime(data['End DateTime'])

    flowers = data['Flower Name'].unique()
    selected_flower = st.selectbox("Select a flower:", flowers)
    
    start_date = st.date_input("Start date:", value=data['Start DateTime'].min().date())
    end_date = st.date_input("End date:", value=data['Start DateTime'].max().date())
    
    filtered_data = data[
        (data['Flower Name'] == selected_flower) &
        (data['Start DateTime'].dt.date >= start_date) &
        (data['Start DateTime'].dt.date <= end_date)
    ]
    
    if filtered_data.empty:
        st.warning("No data available for the selected flower and date range.")
    else:
        filtered_data = filtered_data.sort_values(by='Start DateTime')
        fig = px.line(filtered_data, x='Start DateTime', y='MRP (₹)', title=f"MRP Trend for {selected_flower}",
                      labels={'Start DateTime': 'Date', 'MRP (₹)': 'MRP (₹)'}, markers=True)
        st.plotly_chart(fig, use_container_width=True)

# Predicted Profit for a Date Range
elif feature_option == "Predicted Profit":
    start_date = st.date_input("Select Start Date")
    end_date = st.date_input("Select End Date")
    
    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to the end date.")
        else:
            predicted_profit_dict = get_predicted_profit(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            profit_df = pd.DataFrame(list(predicted_profit_dict.items()), columns=['Flower Name', 'Predicted Profit'])
            st.dataframe(profit_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(profit_df['Flower Name'], profit_df['Predicted Profit'], color='lightgreen', alpha=0.8)
            ax.set_title("Predicted Profit by Flower", fontsize=16)
            ax.set_xlabel("Flower Name", fontsize=12)
            ax.set_ylabel("Predicted Profit", fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(profit_df['Predicted Profit'], labels=profit_df['Flower Name'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.set_title("Profit Distribution by Flower", fontsize=16)
            st.pyplot(fig)
    else:
        st.warning("Please select both start and end dates.")

# Predicted Revenue
elif feature_option == "Predicted Revenue":
    start_date = st.date_input("Select Start Date")
    end_date = st.date_input("Select End Date")
    
    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to the end date.")
        else:
            aggregated_results = get_aggregated_results(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            results_df = pd.DataFrame(list(aggregated_results.items()), columns=['Flower Name', 'Predicted Revenue'])
            st.dataframe(results_df)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(results_df['Predicted Revenue'], labels=results_df['Flower Name'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.set_title("Predicted Revenue Distribution by Flower", fontsize=16)
            st.pyplot(fig)
    else:
        st.warning("Please select both start and end dates.")

# Flower Revenue and Profit Forecast
elif feature_option == "Flower Revenue and Profit Forecast":
    days = st.number_input("Enter the number of days for prediction", min_value=1, max_value=365, value=30)
    
    if days:
        forecast_summary = generate_forecast_summary(days)
        st.dataframe(forecast_summary)

        best_flower_revenue, max_revenue, best_flower_profit, max_profit = find_best_flower(days)
        st.markdown(f"**Best Flower for Revenue:** {best_flower_revenue} (₹{max_revenue:.2f})")
        st.markdown(f"**Best Flower for Profit:** {best_flower_profit} (₹{max_profit:.2f})")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(forecast_summary['Total Predicted Revenue'], labels=forecast_summary['Flower Name'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title("Revenue Distribution by Flower", fontsize=16)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(forecast_summary['Flower Name'], forecast_summary['Total Predicted Profit'], color='lightblue', alpha=0.8)
        ax.set_title("Profit Distribution by Flower", fontsize=16)
        ax.set_xlabel("Flower Name", fontsize=12)
        ax.set_ylabel("Predicted Profit", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

# Predicted Profit for All Flowers
elif feature_option == "Predicted Profit for All Flowers":
    start_date = st.date_input("Select Start Date")
    end_date = st.date_input("Select End Date")
    
    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to the end date.")
        else:
            total_profit_dict = get_total_revenue(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            profit_df = pd.DataFrame(list(total_profit_dict.items()), columns=['Flower Name', 'Predicted Profit'])
            st.dataframe(profit_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(profit_df['Flower Name'], profit_df['Predicted Profit'], color='lightblue', alpha=0.8)
            ax.set_title("Predicted Profit by Flower", fontsize=16)
            ax.set_xlabel("Flower Name", fontsize=12)
            ax.set_ylabel("Predicted Profit", fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(profit_df['Predicted Profit'], labels=profit_df['Flower Name'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.set_title("Profit Distribution by Flower", fontsize=16)
            st.pyplot(fig)
    else:
        st.warning("Please select both start and end dates.")
