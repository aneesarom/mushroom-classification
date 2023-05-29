from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.load_data import LoadData

if __name__ == "__main__":
    load_data = LoadData()
    data_ingestion = DataIngestion()
    data_transformation = DataTransformation()
    load_data.create_database()
    load_data.read_database()
    train_path, test_path = data_ingestion.initialize_data_ingestion()
    train_arr, test_arr, _ = data_transformation.initialize_data_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
