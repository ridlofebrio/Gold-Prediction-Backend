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

CSV_FILE_PATH = "Database/train.csv"

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
            return JSONResponse(
                status_code=400,
                content={
                    "status": {
                        "code": 400,
                        "message": f"Not enough data to form a sequence of size {window_size}. Dataset size: {len(scaled_data)}."
                    },
                    "data": None
                }
            )
        
        # Prepare the last sequence for prediction
        last_sequence = scaled_data[-window_size:]
        
        # Generate predictions for the next 30 days
        predictions = predict_next_n_days(model, last_sequence, scaler, n_days=30)
        
        # Create dates for the predictions
        last_date = df['Tanggal'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        
        # Create prediction results
        prediction_results = []
        for date, price in zip(future_dates, predictions.flatten()):
            prediction_results.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_price": round(float(price), 2)
            })
        
        # Return results with proper structure
        return JSONResponse(
            status_code=200,
            content={
                "status": {
                    "code": 200,
                    "message": "Success"
                },
                "data": {
                    "current_price": {
                        "date": df['Tanggal'].iloc[-1].strftime("%Y-%m-%d"),
                        "price": float(df['Terakhir'].iloc[-1])
                    },
                    "historical_data": df.tail(7)[['Tanggal', 'Terakhir']].apply(
                        lambda x: {
                            'date': x['Tanggal'].strftime("%Y-%m-%d"), 
                            'price': float(x['Terakhir'])
                        }, 
                        axis=1
                    ).tolist(),
                    "predictions": prediction_results,
                    "prediction_summary": {
                        "total_days": 30,
                        "start_date": future_dates[0].strftime("%Y-%m-%d"),
                        "end_date": future_dates[-1].strftime("%Y-%m-%d"),
                        "lowest_price": round(float(min(predictions.flatten())), 2),
                        "highest_price": round(float(max(predictions.flatten())), 2),
                        "average_price": round(float(np.mean(predictions.flatten())), 2)
                    }
                }
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": {
                    "code": 500,
                    "message": f"Internal server error: {str(e)}"
                },
                "data": None
            }
        )
