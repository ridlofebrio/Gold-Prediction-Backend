import os
import logging
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from statsmodels.tsa.vector_ar.vecm import VECMResults

# Define router
router = APIRouter(tags=["vecm"])  # Remove prefix="/vecm"

# Logging configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load model VECM
MODEL_PATH = os.path.join("Model", "vecm_model.pkl")
with open(MODEL_PATH, "rb") as model_file:
    vecm_model = pickle.load(model_file)

# Load and preprocess data
def preprocess_data():
    CSV_FILE_PATH = os.path.join("CSV", "Data Historis Program ML Forecasting Emas(1).csv")
    df = pd.read_csv(CSV_FILE_PATH)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Tanggal'])
    df = df.sort_values(by='Tanggal')
    
    for col in ['Emas', 'Dollar', 'Minyak Dunia']:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    df.set_index('Tanggal', inplace=True)
    return df

# FastAPI app
app = FastAPI()
app.include_router(router)


@router.get("/predict/cagr")
async def predict_cagr(range: str = Query(..., description="Rentang waktu (contoh: 1w, 1m, 3m, 6m)")):
    try:
        # Duration mapping stays the same
        duration = range
        duration_mapping = {
            "1w": {"days": 7, "desc": "1 minggu"},
            "1m": {"days": 30, "desc": "1 bulan"},
            "3m": {"days": 90, "desc": "3 bulan"},
            "6m": {"days": 180, "desc": "6 bulan"}
        }

        if duration not in duration_mapping:
            valid_durations = ", ".join(duration_mapping.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Durasi tidak valid. Pilihan yang tersedia: {valid_durations}"
            )

        n_days = duration_mapping[duration]["days"]

        # Get and validate data
        df = preprocess_data()
        if len(df) < vecm_model.k_ar:
            raise HTTPException(status_code=400, detail="Data historis tidak cukup untuk melakukan prediksi")

        # Get current gold price and date
        current_price = float(df['Emas'].iloc[-1])
        current_date = df.index[-1]

        # Generate predictions
        predictions = vecm_model.predict(steps=n_days)
        
        # Generate future dates
        future_dates = pd.date_range(
            start=current_date + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        
        # Format predictions
        prediction_results = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "price": round(float(pred[0]), 2)
            }
            for date, pred in zip(future_dates, predictions)
        ]
        
        # Calculate summary statistics
        prices = [p['price'] for p in prediction_results]
        prediction_summary = {
            "total_days": len(prediction_results),
            "date_start": round(float(prices[0]), 2),
            "date_end": round(float(prices[-1]), 2),
            "lowest_price": round(float(min(prices)), 2),
            "highest_price": round(float(max(prices)), 2),
            "average_price": round(float(sum(prices) / len(prices)), 2)
        }

        # Return in requested format
        return {
            "close": current_price,
            "date": current_date.strftime("%Y-%m-%d"),
            "prediction": prediction_results,
            "prediction_summary": prediction_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam prediksi: {str(e)}")

