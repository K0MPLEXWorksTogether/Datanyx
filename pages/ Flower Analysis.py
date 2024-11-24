import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# File paths for the dataset and models
dataset_path = "data/flowers_dataset_cleaned.csv"
revenue_model_path = "models/regression/revenue_model_svm.joblib"
profit_model_path = "models/regression/profit_model_svm.joblib"

def analyze_flower(flower_name):
    """
    Analyze a specific flower's revenue and profit forecast.

    Parameters:
    - flower_name (str): The name of the flower to analyze.

    Returns:
    - dict: Summary of the analysis including revenue, profit, and a dataframe of predictions.
    """
    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Encode flower names
    label_encoder = LabelEncoder()
    data['Flower Name'] = label_encoder.fit_transform(data['Flower Name'])
    flower_names = label_encoder.classes_

    if flower_name not in flower_names:
        return {"error": f"'{flower_name}' is not a valid flower name."}

    # Load models
    revenue_model = joblib.load(revenue_model_path)
    profit_model = joblib.load(profit_model_path)

    # Get the encoded value for the selected flower
    flower_encoded = label_encoder.transform([flower_name])[0]

    # Filter dataset for the selected flower
    flower_data = data[data['Flower Name'] == flower_encoded]

    if flower_data.empty:
        return {"error": f"No data available for flower: {flower_name}"}

    # Generate analysis
    flower_results = []
    for _, row in flower_data.iterrows():
        # Create a date range for predictions
        date_range = pd.date_range(start=row['Start DateTime'], end=row['End DateTime'], freq='D')
        freq_qty_sold = np.random.randint(50, 200, len(date_range))  # Random quantities
        average_price = row['MRP (₹)']  # Use MRP for this flower entry

        future_data = pd.DataFrame({
            'Flower Name': [flower_encoded] * len(date_range),
            'Qty Sold (kg)': freq_qty_sold,
            'MRP (₹)': [average_price] * len(date_range)
        })

        # Predict revenue and profit
        future_data['Predicted Revenue'] = revenue_model.predict(future_data[['Flower Name', 'Qty Sold (kg)', 'MRP (₹)']])
        future_data['Predicted Profit'] = profit_model.predict(future_data[['Flower Name', 'Qty Sold (kg)', 'MRP (₹)']])
        future_data['Date'] = date_range

        flower_results.append(future_data)

    # Combine results for the selected flower
    combined_flower_results = pd.concat(flower_results).reset_index(drop=True)
    combined_flower_results['Flower Name'] = flower_name

    # Generate summary
    max_revenue_row = combined_flower_results.loc[combined_flower_results['Predicted Revenue'].idxmax()]
    max_profit_row = combined_flower_results.loc[combined_flower_results['Predicted Profit'].idxmax()]
    max_qty_row = combined_flower_results.loc[combined_flower_results['Qty Sold (kg)'].idxmax()]

    summary = {
        "Flower Name": flower_name,
        "Max Revenue": {
            "Amount": float(max_revenue_row['Predicted Revenue']),
            "Date": str(max_revenue_row['Date'].date())
        },
        "Max Profit": {
            "Amount": float(max_profit_row['Predicted Profit']),
            "Date": str(max_profit_row['Date'].date())
        },
        "Max Quantity Sold": {
            "Quantity": float(max_qty_row['Qty Sold (kg)']),
            "Date": str(max_qty_row['Date'].date())
        }
    }

    return summary

# Streamlit UI components
st.title("Flower Analysis: Revenue and Profit Forecast")
flower_name = st.text_input("Enter the name of the flower to analyze:")

if flower_name:
    result = analyze_flower(flower_name.strip())
    
    if "error" in result:
        st.error(result["error"])
    else:
        # Display the analysis summary
        st.subheader(f"Analysis for {result['Flower Name']}")
        st.write(f"**Max Revenue**: ₹{result['Max Revenue']['Amount']} on {result['Max Revenue']['Date']}")
        st.write(f"**Max Profit**: ₹{result['Max Profit']['Amount']} on {result['Max Profit']['Date']}")
        st.write(f"**Max Quantity Sold**: {result['Max Quantity Sold']['Quantity']} kg on {result['Max Quantity Sold']['Date']}")
        
        # Optionally display the detailed predictions
        st.subheader("Detailed Predictions")
        st.dataframe(pd.DataFrame(result))

