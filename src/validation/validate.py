import pandas as pd
from src.exceptions.data_exceptions import ValidationException


class DataValidator:

    def __init__(self, schema: dict, rules: dict):
        self.schema = schema
        self.rules = rules

    def validate(self, df: pd.DataFrame):
        self._validate_schema(df)
        self._validate_quality(df)

    def _validate_schema(self, df: pd.DataFrame):
        for col, rules in self.schema["columns"].items():
            if rules["required"] and col not in df.columns:
                raise ValidationException(f"Missing required column: {col}")

            if col in df.columns and "allowed_values" in rules:
                if not df[col].isin(rules["allowed_values"]).all():
                    raise ValidationException(f"Invalid values in column: {col}")

    def _validate_quality(self, df: pd.DataFrame):
        if df.isnull().mean().max() > self.rules["missing_values"]["max_missing_ratio"]:
            raise ValidationException("Too many missing values")

        if not self.rules["duplicates"]["allow_duplicates"] and df.duplicated().any():
            raise ValidationException("Duplicate rows found")

        if len(df) < self.rules["row_count"]["min_rows"]:
            raise ValidationException("Insufficient number of rows")
