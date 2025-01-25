import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load model
model = load_model("Model/LSTM.h5")

router = APIRouter()

CSV_FILE_PATH = "CSV/DataTrain.csv"

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

def predict_next_n_days(model, last_sequence, scaler, n_days=180):
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

@router.get("/predict/cagr")
async def predict_cagr(range: str = Query(..., description="Rentang waktu (contoh: 1w, 1m, 3m, 6m)")):
    try:
        # Perbaikan nama variabel untuk konsistensi
        duration = range  # menggunakan nama yang sama dengan parameter query

        # Pemetaan durasi dengan deskripsi yang lebih jelas
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

        # Baca dan preprocessing data
        df = prepare_data()
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="Data tidak ditemukan"
            )

        # Validasi data
        if len(df) < 2:  # minimal butuh 2 data untuk scaling
            raise HTTPException(
                status_code=400,
                detail="Data terlalu sedikit untuk melakukan prediksi"
            )

        try:
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df['Terakhir'].values.reshape(-1, 1))

            # Validasi window size
            window_size = model.input_shape[1]
            if len(scaled_data) < window_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Data tidak cukup untuk membentuk sequence ukuran {window_size}"
                )

            # Prepare sequence dan generate prediksi
            last_sequence = scaled_data[-window_size:]
            predictions = predict_next_n_days(model, last_sequence, scaler, n_days=n_days)

            # Generate tanggal prediksi
            last_date = df['Tanggal'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_days,
                freq='D'
            )

            # Buat hasil prediksi
            prediction_results = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "price": round(float(price), 2)
                }
                for date, price in zip(future_dates, predictions.flatten())
            ]

            # Konversi ke DataFrame untuk analisis
            pred_df = pd.DataFrame(prediction_results)
            pred_df['date'] = pred_df['price'].astype(float)

            # Hitung ringkasan prediksi
            prediction_summary = {
                "total_days": len(pred_df),
                "date_start": pred_df['date'].iloc[0],
                "date_end": pred_df['date'].iloc[-1],
                "lowest_price": round(float(pred_df['price'].min()), 2),
                "highest_price": round(float(pred_df['price'].max()), 2),
                "average_price": round(float(pred_df['price'].mean()), 2)
            }

            # Return hasil yang sama seperti sebelumnya
            return JSONResponse(content={
                "close": float(df['Terakhir'].iloc[-1]),
                "date": df['Tanggal'].iloc[-1].strftime("%Y-%m-%d"),
                "prediction": prediction_results,
                "prediction_summary": prediction_summary
            })

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gagal melakukan prediksi: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan sistem: {str(e)}"
        )
