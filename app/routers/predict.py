import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load model
model = load_model("Model/LSTM.h5")

router = APIRouter()

CSV_FILE_PATH = "Database/DataTrain.csv"

def prepare_data():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Convert date and sort
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Tanggal'])
        df = df.sort_values(by='Tanggal')
        
        # Clean price data
        df['Terakhir'] = df['Terakhir'].str.replace('.', '', regex=False)
        df['Terakhir'] = df['Terakhir'].str.replace(',', '.', regex=False)
        df['Terakhir'] = pd.to_numeric(df['Terakhir'], errors='coerce')
        df = df.dropna(subset=['Terakhir'])
        
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading or processing CSV file: {str(e)}")

def predict_next_n_days(model, last_sequence, scaler, n_days=30):
    predictions = []
    current_sequence = last_sequence.copy()
    window_size = model.input_shape[1]
    
    for _ in range(n_days):
        # Reshape the sequence for prediction
        X_pred = current_sequence.reshape(1, window_size, 1)
        
        # Make prediction
        prediction = model.predict(X_pred, verbose=0)
        predictions.append(prediction[0][0])
        
        # Update the sequence with the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction
    
    # Inverse transform the predictions
    predictions_array = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions_array)

@router.get("/predict")
async def predict_price():
    try:
        # Read and preprocess the file
        df = prepare_data()
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Terakhir'].values.reshape(-1, 1))
        
        # Get window size from model
        window_size = model.input_shape[1]
        
        # Check if dataset is large enough
        if len(scaled_data) < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data to form a sequence of size {window_size}. Dataset size: {len(scaled_data)}."
            )
        
        # Prepare the last sequence for prediction
        last_sequence = scaled_data[-window_size:]
        
        # Generate predictions for the next 7 days
        predictions = predict_next_n_days(model, last_sequence, scaler)
        
        # Create dates for the predictions
        last_date = df['Tanggal'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        
        # Create prediction results
        prediction_results = []
        for date, price in zip(future_dates, predictions.flatten()):
            prediction_results.append({
                "tanggal": date.strftime("%Y-%m-%d"),
                "prediksi_harga": round(float(price), 2)
            })
        
        # Return results
        return JSONResponse(content={
            "last_known_price": float(df['Terakhir'].iloc[-1]),
            "last_known_date": df['Tanggal'].iloc[-1].strftime("%Y-%m-%d"),
            "predictions": prediction_results,
            "historical_data": df.tail(5)[['Tanggal', 'Terakhir']].apply(
                lambda x: {'tanggal': x['Tanggal'].strftime("%Y-%m-%d"), 'harga': float(x['Terakhir'])}, 
                axis=1
            ).tolist()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
