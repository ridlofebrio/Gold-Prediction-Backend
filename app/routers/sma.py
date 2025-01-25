from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

router = APIRouter()

def hitung_sma(data, periode=20):
    return data.rolling(window=periode).mean()

@router.get("/predict/moving-average")
async def sma(range: str = Query(..., description="Rentang waktu (contoh: 1w, 1m, 3m, 6m)")):
    try:
        # Pemetaan durasi untuk rentang data yang akan ditampilkan
        duration_mapping = {
            "1w": {"days": 7},
            "2w": {"days": 14},
            "3w": {"days": 21},
            "1m": {"days": 30},
            "2m": {"days": 60},
            "3m": {"days": 90},
            "4m": {"days": 120},
            "5m": {"days": 150},
            "6m": {"days": 180},
            "7m": {"days": 210},
            "8m": {"days": 240},
            "9m": {"days": 270},
            "10m": {"days": 300},
            "11m": {"days": 330},
            "1y": {"days": 365},
            "2y": {"days": 730},
            "3y": {"days": 1095},
            "4y": {"days": 1460},
            "5y": {"days": 1825}
        }
        
        if range not in duration_mapping:
            valid_durations = ", ".join(duration_mapping.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Durasi tidak valid. Pilihan yang tersedia: {valid_durations}"
            )
        
        days_to_show = duration_mapping[range]["days"]
        
        # Baca data dari CSV
        df = pd.read_csv("CSV/Data 20 tahun.csv")
        
        # Konversi format harga
        df['Terakhir'] = df['Terakhir'].str.replace('.', '', regex=False)
        df['Terakhir'] = df['Terakhir'].str.replace(',', '.', regex=False)
        df['Terakhir'] = pd.to_numeric(df['Terakhir'], errors='coerce')
        
        # Konversi tanggal
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
        df = df.sort_values('Tanggal')
        
        # Hitung SMA dengan periode tetap 20 hari
        sma = hitung_sma(df['Terakhir'])
        
        # Ambil data sesuai rentang waktu yang diminta
        df = df.tail(days_to_show)
        sma = sma.tail(days_to_show)
        
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
            "periode_sma": 20,  # periode tetap 20 hari
            "rentang_waktu": f"{days_to_show} hari",
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
