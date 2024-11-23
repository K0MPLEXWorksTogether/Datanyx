from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Load the revenue model
revenue_model = joblib.load('../models/regression/revenue_model_svm.joblib')

# Load dataset
data = pd.read_csv("../data/flowers_dataset_cleaned.csv")

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

# Initialize Flask app
app = Flask(__name__)

# Flask route to handle the API request
@app.route('/get_aggregated_revenue', methods=['GET'])
def aggregated_revenue():
    # Get start_date and end_date from query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Check if both dates are provided
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date are required"}), 400

    try:
        # Call the function to get aggregated revenue
        aggregated_results = get_aggregated_results(start_date, end_date)
        return jsonify(aggregated_results)

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
