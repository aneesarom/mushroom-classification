import os
import sys
import pandas as pd
from src.exception.exception import CustomException
from src.logging.logging import logging
from dataclasses import dataclass
import pymongo


@dataclass
class LoadDataConfig:
    raw_path = os.path.join("artifacts", "raw.csv")
    input_data_path = os.path.join("notebooks/data", "mushrooms.csv")


class LoadData:
    def __init__(self):
        self.load_data_config = LoadDataConfig()
        self.db = None
        self.collection = None

    def create_database(self):
        try:
            client = pymongo.MongoClient(
                f"mongodb+srv://username:username@cluster0.dxomcpg.mongodb.net/?retryWrites=true&w=majority")
            logging.info("Successfully connected to the MongoDB database.")

            self.db = client["mushroom_classification"]
            self.collection = self.db["mushrooms"]
            df = pd.read_csv(self.load_data_config.input_data_path)
            dict_li = df.to_dict('list')
            self.collection.insert_one(dict_li)
            logging.info("Created the Mushroom Database successfully.")
        except Exception as err:
            raise CustomException(sys, err)

    def read_database(self):
        try:
            document = self.collection.find_one()
            df = pd.DataFrame(document)
            df.drop(["_id"], axis=1, inplace=True)
            logging.info("Successfully retrieved data from MongoDB.")

            os.makedirs(os.path.dirname(self.load_data_config.raw_path), exist_ok=True)
            df.to_csv(self.load_data_config.raw_path)
            logging.info("Raw data has been saved successfully.")
        except Exception as err:
            raise CustomException(sys, err)
