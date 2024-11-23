import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

def predict_flower_prices_by_idx(idx, n_days, model_path="flower_price_predictor_model.keras", dataset_path="hackathon/flowers_dataset_cleaned.csv"):
    """
    Predict flower prices for the next `n_days` based on the flower's index `idx`.

    Arguments:
    - idx: Index of the flower in the dataset (integer).
    - n_days: The number of days to forecast (integer).
    - model_path: Path to the saved .keras model.
    - dataset_path: Path to the flower dataset.

    Returns:
    - A dictionary with flower index and predicted prices.
    """
    
    # Load the .keras model
    model = load_model(model_path)
    
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Debug: Check the first few rows of the dataset
    print("Dataset head:\n", data.head())

    # Load the scalers and label encoder
    scaler_features = joblib.load('scaler_features.joblib')
    scaler_prices = joblib.load('scaler_prices.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    
    # Debug: Check label encoder classes
    print("Label Encoder Classes:", label_encoder.classes_)
    
    # Ensure the index is within the dataset bounds
    if idx < 0 or idx >= len(data):
        return {"error": f"Invalid index {idx}. Please choose a valid index."}
    
    # Get the flower name corresponding to the index
    flower_name = data.iloc[idx]['Flower Name']
    
    # Encode the flower name
    encoded_flower = label_encoder.transform([flower_name])[0]
    print(f"Encoded flower name '{flower_name}' as {encoded_flower}")
    
    # Filter the dataset for the selected flower
    selected_flower_data = data[data['Flower Name'] == flower_name]
    
    # Debug: Check if the flower name is in the dataset
    print(f"Data for flower: {flower_name}\n", selected_flower_data.head())
    
    # Preprocess the data (encode categorical features and normalize)
    features = selected_flower_data.drop(columns=['Start DateTime', 'End DateTime', 'MRP (₹)', 'Flower Name'])
    prices = selected_flower_data['MRP (₹)'].values.reshape(-1, 1)
    
    features_scaled = scaler_features.transform(features)
    prices_scaled = scaler_prices.transform(prices)
    
    # Combine features and prices for sequence generation
    selected_flower_combined = np.hstack([features_scaled, prices_scaled])
    
    # Function to create sequences for LSTM
    def create_sequences(data, seq_length):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :-1])  # Exclude target column (price)
        return np.array(X)
    
    # Create sequences for the selected flower
    sequence_length = 30  # Use the past 30 days to predict the next price
    X_flower = create_sequences(selected_flower_combined, sequence_length)
    
    # Get the last sequence of the flower for prediction
    last_sequence = X_flower[-1]  # Last sequence from the flower's data
    
    # Forecast prices
    def forecast_prices(model, last_sequence, n_days):
        predictions = []
        current_sequence = last_sequence.copy()
        for _ in range(n_days):
            pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)
            predictions.append(pred[0, 0])
            new_step = np.hstack([current_sequence[-1, :-1], pred.flatten()])
            current_sequence = np.vstack([current_sequence[1:], new_step])
        return predictions
    
    # Predict future prices
    predicted_prices_scaled = forecast_prices(model, last_sequence, n_days)
    
    # Rescale predictions to the original price range
    predicted_prices = scaler_prices.inverse_transform([[p] for p in predicted_prices_scaled])[:, 0]
    
    # Return the results as a dictionary
    return {
        "Flower Index": idx,
        "Flower Name": flower_name,
        "Encoded Flower": encoded_flower,
        "Forecasted Prices": [round(float(price), 2) for price in predicted_prices]
    }

# Example usage:
flower_idx = 10  # For example, flower index is 10
n_days = 5  # Forecast for the next 5 days
result = predict_flower_prices_by_idx(flower_idx, n_days)
if "error" in result:
    print(result["error"])
else:
    print(f"\nForecasted Prices for Flower Index {result['Flower Index']} ({result['Flower Name']}):")
    print(f"Encoded Flower: {result['Encoded Flower']}")
    for day, price in enumerate(result["Forecasted Prices"], 1):
        print(f"Day {day}: ₹{price:.2f}")
