import pandas as pd
import mlflow
import mlflow.sklearn
import io

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel


mlflow.set_tracking_uri("file:./mlruns")

MODEL_NAME = "EAP_Attrition_Model"
MODEL_ALIAS = "production"

model = mlflow.sklearn.load_model(
    model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


app = FastAPI(title="EAP Attrition Prediction API")



class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: str
    Department: str
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    Gender: str
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MonthlyIncome: int
    OverTime: str
    TotalWorkingYears: int
    YearsAtCompany: int
    WorkLifeBalance: int



EXPECTED_COLUMNS = [
    "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
    "JobRole", "JobSatisfaction", "MonthlyIncome", "OverTime",
    "TotalWorkingYears", "YearsAtCompany", "WorkLifeBalance"
]



@app.get("/")
def home():
    return {"message": "EAP API is running 🚀"}



@app.post("/predict")
def predict(data: EmployeeInput):
    try:
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability_of_attrition": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

       
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {list(missing)}"
            )

        df_model = df[EXPECTED_COLUMNS]

        predictions = model.predict(df_model)
        probabilities = model.predict_proba(df_model)[:, 1]

        df["prediction"] = predictions
        df["probability_of_attrition"] = probabilities

        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    