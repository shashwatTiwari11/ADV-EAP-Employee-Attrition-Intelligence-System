import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from mlflow.tracking import MlflowClient


class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self):
        return pd.read_csv(self.dataset_path)


class FeatureAnalyzer:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def clean(self):
        drop_cols = [
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ]
        self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])
        return self.df

    def split_features_target(self):
        if self.df[self.target_column].dtype == "object":
            self.df[self.target_column] = self.df[self.target_column].map({"Yes": 1, "No": 0})

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        categorical = X.select_dtypes(include=["object"]).columns.tolist()
        numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

        feature_schema = {
            "numerical": numerical,
            "categorical": categorical,
            "target": self.target_column
        }

        return X, y, feature_schema


class PreprocessingBuilder:
    @staticmethod
    def build(numerical_cols, categorical_cols):
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ]
        )


class ModelFactory:
    @staticmethod
    def get_models(model_type):
        models = {
            "logistic": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(n_estimators=200, random_state=42)
        }

        if model_type == "auto":
            return models
        else:
            return {model_type: models[model_type]}


class ModelEvaluator:
    @staticmethod
    def evaluate(pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }


class MetadataManager:
    @staticmethod
    def log(feature_schema, training_metadata):
        mlflow.log_dict(feature_schema, "feature_schema.json")
        mlflow.log_dict(training_metadata, "training_metadata.json")


class RegistryManager:
    @staticmethod
    def promote_to_production(model_name):
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name)[0].version

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=latest_version
        )

        return latest_version


class ModelTrainer:
    def __init__(self, target_column: str, model_type: str = "auto"):
        self.target_column = target_column
        self.model_type = model_type

    def train(self, dataset_path: str):

        # === Load Data ===
        df = DataLoader(dataset_path).load()

        # === Feature Engineering ===
        analyzer = FeatureAnalyzer(df, self.target_column)
        df = analyzer.clean()
        X, y, feature_schema = analyzer.split_features_target()

        # === Train/Test Split ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # === Build Preprocessor ===
        preprocessor = PreprocessingBuilder.build(
            feature_schema["numerical"],
            feature_schema["categorical"]
        )

        # === Model Selection ===
        models = ModelFactory.get_models(self.model_type)

        best_model_name = None
        best_pipeline = None
        best_auc = 0

        for name, model in models.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

            if auc > best_auc:
                best_auc = auc
                best_model_name = name
                best_pipeline = pipeline

        # === Evaluate Best Model ===
        metrics = ModelEvaluator.evaluate(best_pipeline, X_test, y_test)

        mlflow.set_experiment("EAP_Attrition_Experiment")

        with mlflow.start_run():

            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_param("model_type_input", self.model_type)

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            training_metadata = {
                "dataset_rows": len(df),
                "dataset_columns": list(df.columns),
                "candidate_models": list(models.keys()),
                "selected_model": best_model_name
            }

            MetadataManager.log(feature_schema, training_metadata)

            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path="model",
                registered_model_name="EAP_Attrition_Model"
            )

        version = RegistryManager.promote_to_production("EAP_Attrition_Model")

        return {
            **metrics,
            "model_type": best_model_name,
            "model_version": version
        }