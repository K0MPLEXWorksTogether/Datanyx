from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Load the dataset and models
data = pd.read_csv("../../data/flowers_dataset_cleaned.csv")

if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

label_encoder = LabelEncoder()
data['Flower Name Encoded'] = label_encoder.fit_transform(data['Flower Name'])
flower_names = label_encoder.classes_

revenue_model = joblib.load('../../models/regression/revenue_model_svm.joblib')
profit_model = joblib.load('../../models/regression/profit_model_svm.joblib')

# Function to get total revenue based on the start and end dates
def get_total_revenue(start_date: str, end_date: str):
   
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    results = []

    for flower in flower_names:
        encoded_flower = label_encoder.transform([flower])[0] 
        freq_qty_sold = np.random.randint(50, 200, len(date_range))  
        average_price = data['MRP (₹)'].mean()  

        future_data = pd.DataFrame({
            'Flower Name': [encoded_flower] * len(date_range),
            'Qty Sold (kg)': freq_qty_sold,
            'MRP (₹)': [average_price] * len(date_range)
        })

        predicted_revenue = revenue_model.predict(future_data)

        for i, single_date in enumerate(date_range):
            results.append({
                'Flower Name': flower,
                'Date': single_date,
                'Predicted Revenue': predicted_revenue[i]
            })

    prediction_summary = pd.DataFrame(results)

    aggregated_revenue = prediction_summary.groupby('Flower Name')['Predicted Revenue'].sum().reset_index()

    top_revenue = aggregated_revenue.sort_values(by='Predicted Revenue', ascending=False)

    revenue_dict = dict(zip(top_revenue['Flower Name'], top_revenue['Predicted Revenue']))

    return revenue_dict

# Initialize Flask app
app = Flask(__name__)

# Flask route to handle the API request
@app.route('/get_total_revenue', methods=['GET'])
def total_revenue():
    # Get start_date and end_date from query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Check if both dates are provided
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date are required"}), 400

    try:
        # Call the function to get total revenue
        revenue_dict = get_total_revenue(start_date, end_date)
        return jsonify(revenue_dict)

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5003)
