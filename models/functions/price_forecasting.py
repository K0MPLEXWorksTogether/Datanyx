import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

revenue_model = joblib.load('models/regression/revenue_model_svm.joblib')
profit_model = joblib.load('models/regression/profit_model_svm.joblib')

data = pd.read_csv('data/flowers_dataset_cleaned.csv')

label_encoder = LabelEncoder()
data['Flower Name'] = label_encoder.fit_transform(data['Flower Name'])

flower_names = label_encoder.classes_

def predict_for_days(flower_name, days):
    encoded_flower = label_encoder.transform([flower_name])[0]
    freq_qty_sold = np.random.randint(50, 200, days)
    average_price = data['MRP (₹)'].mean()

    future_data = pd.DataFrame({
        'Flower Name': [encoded_flower] * days,
        'Qty Sold (kg)': freq_qty_sold,
        'MRP (₹)': [average_price] * days
    })

    predicted_revenue = revenue_model.predict(future_data[['Flower Name', 'Qty Sold (kg)', 'MRP (₹)']])
    predicted_profit = profit_model.predict(future_data[['Flower Name', 'Qty Sold (kg)', 'MRP (₹)']])
    
    total_revenue = predicted_revenue.sum()
    total_profit = predicted_profit.sum()

    return total_revenue, total_profit

def find_best_flower(days):
    max_revenue = -np.inf
    max_profit = -np.inf
    best_flower_revenue = None
    best_flower_profit = None

    for flower in flower_names:
        revenue, profit = predict_for_days(flower, days)
        if revenue > max_revenue:
            max_revenue = revenue
            best_flower_revenue = flower
        if profit > max_profit:
            max_profit = profit
            best_flower_profit = flower
    
    return best_flower_revenue, max_revenue, best_flower_profit, max_profit

def generate_forecast_summary(days):
    results = []
    for flower in flower_names:
        total_revenue, total_profit = predict_for_days(flower, days)
        results.append({
            "Flower Name": flower,
            "Total Predicted Revenue": total_revenue,
            "Total Predicted Profit": total_profit
        })
    forecast_summary = pd.DataFrame(results)
    return forecast_summary

days = int(input("Enter the number of days you want to predict for: "))

best_flower_revenue, max_revenue, best_flower_profit, max_profit = find_best_flower(days)

forecast_summary = generate_forecast_summary(days)
print("\nForecast Summary:")
print(forecast_summary)

print(f"\nFor {days} days, the flower with the highest predicted revenue is '{best_flower_revenue}' with ₹{max_revenue:.2f}")
print(f"The flower with the highest predicted profit is '{best_flower_profit}' with ₹{max_profit:.2f}")
