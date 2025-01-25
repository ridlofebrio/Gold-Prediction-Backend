import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def get_gold_price_history():
    try:
        # Dapatkan tanggal hari ini dan 4 bulan yang lalu
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download data harga emas menggunakan yfinance
        gold = yf.download("GC=F", start=start_date, end=end_date)
        
        # Format data
        gold = gold.reset_index()
        gold['Date'] = gold['Date'].dt.strftime('%d/%m/%Y')
        
        # Hitung persentase perubahan terlebih dahulu
        pct_change = (gold['Close'].pct_change() * 100).round(2)
        
        # Format angka dengan koma sebagai pemisah desimal
        def format_number(value):
            return '{:,.2f}'.format(value).replace(',', 'X').replace('.', ',').replace('X', '.')
        
        # Konversi kolom numerik menggunakan numpy vectorize
        format_vec = np.vectorize(format_number)
        gold['Close'] = format_vec(gold['Close'].values)
        gold['Open'] = format_vec(gold['Open'].values)
        gold['High'] = format_vec(gold['High'].values)
        gold['Low'] = format_vec(gold['Low'].values)
        
        # Rename kolom
        gold = gold.rename(columns={
            'Date': 'Tanggal',
            'Close': 'Terakhir', 
            'Open': 'Pembukaan',
            'High': 'Tertinggi',
            'Low': 'Terendah',
            'Volume': 'Vol.',
        })
        
        # Tambahkan persentase perubahan yang sudah dihitung
        gold['Perubahan%'] = pct_change.astype(str) + '%'
        
        # Format volume dalam ribuan (K)
        gold['Vol.'] = (gold['Vol.'] / 1000).round(2).astype(str) + 'K'
        
        # Pilih dan urutkan kolom
        gold = gold[['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%']]
        
        # Simpan ke CSV
        gold.to_csv('CSV/DataTrain.csv', index=False, quoting=1)
        return True
        
    except Exception as e:
        print(f"Error mengambil data harga emas: {str(e)}")
        return False

def should_update():
    """Cek apakah sekarang waktunya untuk update (jam 19:00)"""
    now = datetime.now()
    return now.hour == 19 and now.minute == 0

def periodic_update():
    while True:
        if should_update():
            # Update data harian
            success_daily = get_gold_price_history()
            if success_daily:
                print(f"Data harian berhasil diupdate pada {datetime.now()}")
            else:
                print(f"Gagal mengupdate data harian pada {datetime.now()}")
                
            # Update data historis 20 tahun
            success_historical = get_historical_gold_data()
            if success_historical:
                print(f"Data historis 20 tahun berhasil diupdate pada {datetime.now()}")
            else:
                print(f"Gagal mengupdate data historis 20 tahun pada {datetime.now()}")
                
        # Tunggu 60 detik sebelum pengecekan berikutnya
        time.sleep(60)

def get_historical_gold_data():
    try:
        # Dapatkan tanggal hari ini dan 20 tahun yang lalu
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*20)
        
        # Download data harga emas menggunakan yfinance
        gold = yf.download("GC=F", start=start_date, end=end_date)
        
        # Format data
        gold = gold.reset_index()
        gold['Date'] = gold['Date'].dt.strftime('%d/%m/%Y')
        
        # Hitung persentase perubahan
        pct_change = (gold['Close'].pct_change() * 100).round(2)
        
        # Format angka dengan koma sebagai pemisah desimal
        def format_number(value):
            return '{:,.2f}'.format(value).replace(',', 'X').replace('.', ',').replace('X', '.')
        
        # Konversi kolom numerik menggunakan numpy vectorize
        format_vec = np.vectorize(format_number)
        gold['Close'] = format_vec(gold['Close'].values)
        gold['Open'] = format_vec(gold['Open'].values)
        gold['High'] = format_vec(gold['High'].values)
        gold['Low'] = format_vec(gold['Low'].values)
        
        # Rename kolom
        gold = gold.rename(columns={
            'Date': 'Tanggal',
            'Close': 'Terakhir', 
            'Open': 'Pembukaan',
            'High': 'Tertinggi',
            'Low': 'Terendah',
            'Volume': 'Vol.',
        })
        
        # Tambahkan persentase perubahan
        gold['Perubahan%'] = pct_change.astype(str) + '%'
        
        # Format volume dalam ribuan (K)
        gold['Vol.'] = (gold['Vol.'] / 1000).round(2).astype(str) + 'K'
        
        # Pilih dan urutkan kolom
        gold = gold[['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%']]
        
        # Simpan ke CSV
        gold.to_csv('CSV/Data 20 tahun.csv', index=False, quoting=1)
        return True
        
    except Exception as e:
        print(f"Error mengambil data historis emas: {str(e)}")
        return False
