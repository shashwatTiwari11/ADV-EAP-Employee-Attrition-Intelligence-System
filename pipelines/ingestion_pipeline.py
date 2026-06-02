from datetime import datetime
from pathlib import Path

from src.ingestion.ingest import CSVIngestor
from src.validation.validate import DataValidator
from src.utils.config_loader import ConfigLoader


def run_ingestion_pipeline(uploaded_file_path: str):
    schema = ConfigLoader.load_config("config/schema.yaml")
    rules = ConfigLoader.load_config("config/validation.yaml")

    df = CSVIngestor().ingest(uploaded_file_path)

    validator = DataValidator(schema, rules)
    validator.validate(df)

    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ref_dir / f"reference_{timestamp}.csv"

    df.to_csv(output_path, index=False)

    return str(output_path)
