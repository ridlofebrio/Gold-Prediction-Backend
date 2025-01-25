from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

router = APIRouter()

def hitung_sma(data, periode):
    return data.rolling(window=periode).mean()

@router.get("/sma/{periode}")
async def get_sma(periode: int):
    try:
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
