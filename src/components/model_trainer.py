import os
import sys
from src.logging.logging import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split
from src.utils.utils import save_object
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_pickle_path = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            x_train, x_test, y_train, y_test = train_arr[:, :-1], test_arr[:, :-1], train_arr[:, -1], test_arr[:, -1]
            logging.info(f"train_test_shape: \n{x_train.shape, y_train.shape, x_test.shape, y_test.shape}")

            models = {
                "dt": DecisionTreeClassifier(random_state=42),
                "rf": RandomForestClassifier(random_state=42)
            }

            f1_macro_list = []
            logging.info("Model training has been successfully initiated")

            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = f1_score(y_test, y_pred, average='macro')
                f1_macro_list.append(score)

            logging.info("Model training has been successfully completed")

            max_value = max(f1_macro_list)
            max_value_index = f1_macro_list.index(max_value)
            best_model_name = list(models.keys())[max_value_index]
            best_model = list(models.values())[max_value_index]
            logging.info(f"Best_model: {best_model_name}, F1 macro: {max_value}")

            save_object(self.model_trainer_config.model_pickle_path, best_model)
            logging.info("Best model was successfully saved")

            return f1_macro_list

        except Exception as err:
            raise CustomException(sys, err)