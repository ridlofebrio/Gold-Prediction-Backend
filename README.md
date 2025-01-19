# Gold-Prediction-Backend

Backend API untuk prediksi harga emas menggunakan model LSTM.

## Prasyarat

Sebelum menginstal, pastikan sistem Anda memiliki:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

## Instalasi

1. Clone repositori ini
2. Buat virtual environment (opsional tapi disarankan)

3. Aktifkan virtual environment
- Untuk Windows:
```bash
venv\Scripts\activate
```
- Untuk Linux/Mac:
```bash
source venv/bin/activate
```

4. Instal dependensi yang diperlukan
```bash
pip install -r requirements.txt
```

## Struktur Data

Pastikan Anda memiliki file-file berikut di direktori yang sesuai:
- `Database/Data 20 tahun.csv` - Dataset historis harga emas
- `Database/train.csv` - Data training untuk model
- `Model/LSTM.h5` - Model LSTM yang sudah dilatih

## Menjalankan Aplikasi

1. Jalankan server FastAPI menggunakan uvicorn:
```bash
uvicorn app.main:app --reload
```

2. Server akan berjalan di `http://localhost:8000`

## Endpoint API

- `GET /data` - Mendapatkan seluruh data historis
- `GET /columns` - Mendapatkan daftar kolom dataset
- `GET /predict` - Mendapatkan prediksi harga emas untuk 7 hari ke depan

## Dokumentasi API

Setelah menjalankan server, Anda dapat mengakses:
- Dokumentasi Swagger UI: `http://localhost:8000/docs`
- Dokumentasi ReDoc: `http://localhost:8000/redoc`

## Teknologi yang Digunakan

- FastAPI
- TensorFlow
- Pandas
- Scikit-learn
- NumPy

## Lisensi

[Tambahkan informasi lisensi di sini]