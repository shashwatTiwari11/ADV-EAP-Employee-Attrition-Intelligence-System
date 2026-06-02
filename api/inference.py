import joblib
import pandas as pd
from pathlib import Path


class InferenceService:

    def __init__(self, model_path: str, preprocessor_path: str, threshold: float = 0.5):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.threshold = threshold

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if not Path(self.preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, input_data: dict):

        df = pd.DataFrame([input_data])

        transformed = self.preprocessor.transform(df)

        probability = self.model.predict_proba(transformed)[:, 1][0]

        prediction = int(probability >= self.threshold)

        return {
            "probability_of_attrition": float(probability),
            "prediction": prediction,
            "threshold_used": self.threshold
        }
