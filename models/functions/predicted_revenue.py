import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Load the revenue model
revenue_model = joblib.load('models/regression/revenue_model_svm.joblib')

# Load dataset
data = pd.read_csv("data/flowers_dataset_cleaned.csv")

# Encode flower names
label_encoder = LabelEncoder()
data['Flower Name Encoded'] = label_encoder.fit_transform(data['Flower Name'])
flower_names = label_encoder.classes_

# Function to get aggregated results
def get_aggregated_results(start_date: str, end_date: str):
    # Convert string dates to datetime
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    results = []

    for flower in flower_names:
        # Get encoded value for flower
        encoded_flower = label_encoder.transform([flower])[0]

        # Randomly generate quantities for each day
        freq_qty_sold = np.random.randint(50, 200, len(date_range))
        average_price = data['MRP (₹)'].mean()

        # Prepare future data for predictions
        future_data = pd.DataFrame({
            'Flower Name': [encoded_flower] * len(date_range),
            'Qty Sold (kg)': freq_qty_sold,
            'MRP (₹)': [average_price] * len(date_range)
        })

        # Predict revenue
        predicted_revenue = revenue_model.predict(future_data)

        # Collect results
        for i, single_date in enumerate(date_range):
            results.append({
                'Flower Name': flower,
                'Date': single_date,
                'Predicted Revenue': predicted_revenue[i]
            })

    # Create a DataFrame with all the results
    prediction_summary = pd.DataFrame(results)

    # Aggregate results by flower name
    aggregated_results = prediction_summary.groupby('Flower Name').agg({
        'Predicted Revenue': 'sum'
    }).reset_index()

    # Convert the aggregated results to a dictionary
    aggregated_results_dict = aggregated_results.set_index('Flower Name').to_dict()['Predicted Revenue']
    
    return aggregated_results_dict


# Example usage:
start_date = "2024-11-01"
end_date = "2024-11-30"
aggregated_results = get_aggregated_results(start_date, end_date)

# Print aggregated results
print(aggregated_results)
