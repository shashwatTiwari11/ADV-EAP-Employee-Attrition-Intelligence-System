import pandas as pd

from src.exceptions.data_exceptions import IngestionException


class CSVIngestor:
    def ingest(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise IngestionException(f"CSV ingestion failed: {str(e)}")
