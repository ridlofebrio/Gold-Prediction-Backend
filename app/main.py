from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
from app.routers import predict
from app.routers import sma
from app.routers import vecm
from app.schedulers.data_updater import get_gold_price_history, should_update, get_historical_gold_data
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
import time

app = FastAPI()

# Tambahkan CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include predict router
# app.include_router(predict.router)
app.include_router(sma.router)
app.include_router(vecm.router)

# Fungsi untuk menjalankan update data secara periodik
def periodic_update():
    while True:
        get_gold_price_history()
        get_historical_gold_data()
        time.sleep(3600)

# Event handler saat aplikasi startup
@app.on_event("startup")
async def startup_event():
    # Cek apakah perlu update saat startup
    if should_update():
        get_gold_price_history()
        get_historical_gold_data()
    
    # Jalankan update periodik dalam thread terpisah
    update_thread = threading.Thread(target=periodic_update, daemon=True)
    update_thread.start()

csv_file_path = "CSV/DataTrain.csv"
data = pd.read_csv(csv_file_path)

# Mengganti NaN dengan None agar JSON compliant
data = data.where(pd.notnull(data), None)

@app.get("/columns")
async def get_columns():
    return JSONResponse(content={"columns": data.columns.tolist()})

@app.get("/data")
async def get_all_data():
    formatted_data = {
        "status": "success",
        "total_records": len(data),
        "data": data.to_dict(orient="records")
    }
    return JSONResponse(
        content=formatted_data,
        status_code=200
    )

