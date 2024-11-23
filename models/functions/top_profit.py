import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/flowers_dataset_cleaned.csv")

if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

label_encoder = LabelEncoder()
data['Flower Name Encoded'] = label_encoder.fit_transform(data['Flower Name'])
flower_names = label_encoder.classes_

revenue_model = joblib.load('models/regression/revenue_model_svm.joblib')
profit_model = joblib.load('models/regression/profit_model_svm.joblib')

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

        predicted_revenue = profit_model.predict(future_data)

        for i, single_date in enumerate(date_range):
            results.append({
                'Flower Name': flower,
                'Date': single_date,
                'Predicted Profit': predicted_revenue[i]
            })

    prediction_summary = pd.DataFrame(results)

    aggregated_revenue = prediction_summary.groupby('Flower Name')['Predicted Profit'].sum().reset_index()

    top_revenue = aggregated_revenue.sort_values(by='Predicted Profit', ascending=False)

    revenue_dict = dict(zip(top_revenue['Flower Name'], top_revenue['Predicted Profit']))

    return revenue_dict

start_date = "2024-11-01"
end_date = "2024-11-10"
revenue_dict = get_total_revenue(start_date, end_date)

print(revenue_dict)