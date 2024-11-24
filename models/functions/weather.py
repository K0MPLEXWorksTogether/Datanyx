import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_models():
    rf_model = joblib.load("flower_name_predictor_rf.pkl")
    encoder = joblib.load("categorical_encoder.pkl")
    scaler = joblib.load("numerical_scaler.pkl")
    flower_encoder = joblib.load("flower_label_encoder.pkl")
    return rf_model, encoder, scaler, flower_encoder

def predict_flower(weather, qty_sold, mrp, customer_segment):
    rf_model, encoder, scaler, flower_encoder = load_models()

    input_data = pd.DataFrame({
        'Weather': [weather],
        'Qty Sold (kg)': [qty_sold],
        'MRP (₹)': [mrp],
        'Customer Segment': [customer_segment]
    })
    
    input_data['Revenue'] = input_data['MRP (₹)'] * input_data['Qty Sold (kg)']
    
    categorical_features = ['Weather', 'Customer Segment']
    numerical_features = ['MRP (₹)', 'Qty Sold (kg)', 'Revenue']

    X_categorical = encoder.transform(input_data[categorical_features])

    X_numerical = scaler.transform(input_data[numerical_features])

    X_combined = np.hstack((X_categorical, X_numerical))

    predicted_class = rf_model.predict(X_combined)

    predicted_flower = flower_encoder.inverse_transform(predicted_class)
    
    return predicted_flower[0]

weather = 'Sunny'  
qty_sold = 5  
mrp = 50  
customer_segment = 'Marriage'  

predicted_flower_name = predict_flower(weather, qty_sold, mrp, customer_segment)
print(f"The predicted flower name is: {predicted_flower_name}")
