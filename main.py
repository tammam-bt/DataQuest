"""
FastAPI backend for the insurance coverage prediction pipeline.
Uses solution.load_model(), solution.preprocess(), and solution.predict() without modification.
"""
import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

import solution

app = FastAPI(
    title="Insurance Coverage Prediction API",
    description="Upload a CSV to get predicted Purchased_Coverage_Bundle per User_ID.",
    version="1.0.0",
)

# Model loaded once at startup
model = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.on_event("startup")
def load_model_on_startup():
    global model
    model = solution.load_model()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    try:
        df_preprocessed = solution.preprocess(df)
        predictions_df = solution.predict(df_preprocessed, model)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing or prediction failed: {str(e)}")
    # Return list of dicts with User_ID and Purchased_Coverage_Bundle
    return predictions_df.to_dict(orient="records")


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found.")
    return FileResponse(index_path)
