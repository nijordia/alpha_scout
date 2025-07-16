from typing import Any, Dict
import pandas as pd

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_data(self) -> pd.DataFrame:
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        return self.data

    def transform_data(self) -> pd.DataFrame:
        # Example transformation: converting date column to datetime
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
        return self.data

    def process_data(self) -> pd.DataFrame:
        cleaned_data = self.clean_data()
        transformed_data = self.transform_data()
        return transformed_data

    def get_processed_data(self) -> Dict[str, Any]:
        processed_data = self.process_data()
        return {
            "data": processed_data,
            "summary": {
                "num_rows": processed_data.shape[0],
                "num_columns": processed_data.shape[1]
            }
        }