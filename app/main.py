from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
from app.routers import predict

app = FastAPI()

# Include predict router
app.include_router(predict.router)

csv_file_path = "Database/Data 20 tahun.csv"
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

