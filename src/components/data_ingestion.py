import os
import sys
import pandas as pd
from src.exception.exception import CustomException
from src.logging.logging import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_path = os.path.join("artifacts", "raw.csv")
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")
    input_data_path = os.path.join("notebooks/data", "mushrooms.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        try:
            logging.info("Data ingestion has been initiated")
            df = pd.read_csv(self.ingestion_config.input_data_path)
            logging.info("Successfully read data from the source data")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path)
            logging.info("Raw data has been successfully created")

            train_set, test_set = train_test_split(df, random_state=42, test_size=0.3)
            logging.info("Train and Test data has been successfully split")
            train_set.to_csv(self.ingestion_config.train_path)
            test_set.to_csv(self.ingestion_config.test_path)
            logging.info("Data ingestion has been successfully completed")

            return self.ingestion_config.train_path, self.ingestion_config.test_path
        except Exception as err:
            raise CustomException(sys, err)
