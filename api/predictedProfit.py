from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

# Load the pre-saved prediction summary
prediction_summary = joblib.load('../models/regression/prediction_summary.joblib')

# Function to get predicted profit as a dictionary based on start_date and end_date
def get_predicted_profit(start_date: str, end_date: str):
    # Convert string dates to datetime
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Ensure that 'Date' column is in datetime format
    prediction_summary['Date'] = pd.to_datetime(prediction_summary['Date'])

    # Filter the data based on the date range
    filtered_data = prediction_summary[(prediction_summary['Date'] >= start_date) & (prediction_summary['Date'] <= end_date)]

    # Create a dictionary with 'Flower Name' as the key and 'Predicted Profit' as the value
    profit_dict = {row['Flower Name']: row["Predicted Profit"]
                   for _, row in filtered_data.iterrows()}
    return profit_dict

# Initialize Flask app
app = Flask(__name__)

# Flask route to handle the API request
@app.route('/get_predicted_profit', methods=['GET'])
def predicted_profit():
    # Get start_date and end_date from query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Check if both dates are provided
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date are required"}), 400

    try:
        # Call the function to get predicted profits
        profit_dict = get_predicted_profit(start_date, end_date)
        return jsonify(profit_dict)

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
