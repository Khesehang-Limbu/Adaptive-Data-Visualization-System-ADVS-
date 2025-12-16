from pathlib import Path

import pandas as pd
from pandas import DataFrame


class DatasetLoader:
    @staticmethod
    def load_from_file(file_path: str) -> DataFrame | ValueError:
        """
        Load DataFrame from file path.

        :param file_path:
        :return: DataFrame
        """
        file_path = Path(file_path)

        if file_path.suffix == ".csv":
            return pd.read_csv(file_path, sep=None, engine="python")
        elif file_path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            return ValueError(f"Unsupported file format: {file_path.suffix}")

    @staticmethod
    def load_from_model(dataset_model) -> pd.DataFrame:
        """
        Load DataFrame from Django Model.
        :param dataset_model:
        :return: DataFrame
        """
        return DatasetLoader.load_from_file(dataset_model.file.path)
