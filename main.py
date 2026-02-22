"""
FastAPI backend for the insurance coverage prediction pipeline.
Uses solution.load_model(), solution.preprocess(), and solution.predict() without modification.
"""
import io
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import solution

# --- Manual prediction request body (must match solution.preprocess input columns) ---
BrokerAgencyType = Literal["National_Corporate", "Urban_Boutique"]
DeductibleTier = Literal["Tier_1_High_Ded", "Tier_2_Mid_Ded", "Tier_3_Low_Ded", "Tier_4_Zero_Ded"]
AcquisitionChannel = Literal["Affiliate_Group", "Aggregator_Site", "Corporate_Partner", "Direct_Website", "Local_Broker"]
PaymentSchedule = Literal["Annual_Upfront", "Monthly_EFT", "Quarterly_Invoice"]
EmploymentStatus = Literal["Contractor", "Employed_FullTime", "Self_Employed", "Unemployed"]
PolicyStartMonth = Literal["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


class ManualPredictionRequest(BaseModel):
    broker_id: str = Field(..., alias="Broker_ID", description="Broker identifier")
    region_code: str = Field(..., alias="Region_Code", description="Region code")
    broker_agency_type: BrokerAgencyType = Field(..., alias="Broker_Agency_Type")
    deductible_tier: DeductibleTier = Field(..., alias="Deductible_Tier")
    acquisition_channel: AcquisitionChannel = Field(..., alias="Acquisition_Channel")
    payment_schedule: PaymentSchedule = Field(..., alias="Payment_Schedule")
    employment_status: EmploymentStatus = Field(..., alias="Employment_Status")
    policy_start_month: PolicyStartMonth = Field(..., alias="Policy_Start_Month")
    adult_dependents: int = Field(..., alias="Adult_Dependents", ge=0)
    child_dependents: int = Field(..., alias="Child_Dependents", ge=0)
    infant_dependents: int = Field(..., alias="Infant_Dependents", ge=0)
    estimated_annual_income: float = Field(..., alias="Estimated_Annual_Income", ge=0)
    previous_claims_filed: int = Field(..., alias="Previous_Claims_Filed", ge=0)
    previous_policy_duration_months: int = Field(..., alias="Previous_Policy_Duration_Months", ge=0)

    class Config:
        populate_by_name = True

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


@app.post("/predict/manual")
def predict_manual(body: ManualPredictionRequest):
    """Accept a single record as JSON, preprocess and predict, return Purchased_Coverage_Bundle."""
    row = body.model_dump(by_alias=True)
    row["User_ID"] = "manual_1"
    df = pd.DataFrame([row])
    try:
        df_preprocessed = solution.preprocess(df)
        predictions_df = solution.predict(df_preprocessed, model)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing or prediction failed: {str(e)}")
    bundle = int(predictions_df["Purchased_Coverage_Bundle"].iloc[0])
    return {"Purchased_Coverage_Bundle": bundle}


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found.")
    return FileResponse(index_path)
