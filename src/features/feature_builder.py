import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from src.features.transformers import build_preprocessor


class FeatureBuilder:

    def __init__(self, target_column: str, drop_columns: list = None):
        self.target_column = target_column
        self.drop_columns = drop_columns or []

    def build(self, reference_csv_path: str):

        if not Path(reference_csv_path).exists():
            raise FileNotFoundError(f"Reference file not found: {reference_csv_path}")

        df = pd.read_csv(reference_csv_path)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

        # Drop unwanted columns if present
        df = df.drop(columns=[col for col in self.drop_columns if col in df.columns])

        # Convert Yes/No target to 1/0 if applicable
        if df[self.target_column].dtype == "object":
            df[self.target_column] = df[self.target_column].map(
                {"Yes": 1, "No": 0}
            )

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Auto-detect column types
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        preprocessor = build_preprocessor(numerical_cols, categorical_cols)

        X_transformed = preprocessor.fit_transform(X)

        # Build feature names dynamically
        cat_feature_names = []
        if categorical_cols:
            cat_feature_names = list(
                preprocessor.named_transformers_["cat"]
                .get_feature_names_out(categorical_cols)
            )

        feature_names = numerical_cols + cat_feature_names

        features_df = pd.DataFrame(X_transformed, columns=feature_names)
        features_df[self.target_column] = y.values

        # Save preprocessor
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

        # Save processed dataset
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/processed/features_{timestamp}.csv"

        features_df.to_csv(output_path, index=False)

        return output_path
