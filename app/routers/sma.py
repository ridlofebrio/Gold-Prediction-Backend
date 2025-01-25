from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

router = APIRouter()

def hitung_sma(data, periode):
    return data.rolling(window=periode).mean()

@router.get("/predict/moving-average")
async def sma(range: str = Query(..., description="Rentang waktu (contoh: 1w, 1m, 3m, 6m)")):
    try:
        # Pemetaan durasi ke periode SMA
        duration_mapping = {
            "1w": {"days": 7, "periode": 7},
            "1m": {"days": 30, "periode": 30},
            "3m": {"days": 90, "periode": 90},
            "6m": {"days": 180, "periode": 180}
        }
        
        if range not in duration_mapping:
            valid_durations = ", ".join(duration_mapping.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Durasi tidak valid. Pilihan yang tersedia: {valid_durations}"
            )
        
        periode = duration_mapping[range]["periode"]
        
        # Baca data dari CSV yang sama
        df = pd.read_csv("CSV/Data 20 tahun.csv")
        
        # Konversi format harga
        df['Terakhir'] = df['Terakhir'].str.replace('.', '', regex=False)
        df['Terakhir'] = df['Terakhir'].str.replace(',', '.', regex=False)
        df['Terakhir'] = pd.to_numeric(df['Terakhir'], errors='coerce')
        
        # Konversi tanggal
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
        df = df.sort_values('Tanggal')
        
        # Hitung SMA
        sma = hitung_sma(df['Terakhir'], periode)
        
        # Siapkan hasil
        hasil = []
        for tanggal, harga, nilai_sma in zip(df['Tanggal'], df['Terakhir'], sma):
            if pd.notna(nilai_sma):  # Hanya masukkan data yang tidak NaN
                hasil.append({
                    "tanggal": tanggal.strftime("%Y-%m-%d"),
                    "harga": round(float(harga), 2),
                    "sma": round(float(nilai_sma), 2)
                })
        
        # Hitung statistik
        statistik = {
            "periode_sma": periode,
            "rata_rata_sma": round(float(sma.mean()), 2),
            "sma_terakhir": round(float(sma.iloc[-1]), 2),
            "total_data": len(hasil)
        }
        
        return JSONResponse(content={
            "status": "success",
            "statistik": statistik,
            "data": hasil
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
