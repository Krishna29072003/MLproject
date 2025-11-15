import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("===== Data Ingestion Started =====")
        try:
            # Reading dataset
            logging.info("Reading dataset from data/stud.csv")
            #df = pd.read_csv('notebook\data\stud.csv')
            df = pd.read_csv('notebook/data/stud.csv')

            logging.info("Dataset loaded successfully")

            # Create artifacts folder
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            logging.info("Artifacts folder created (if not present)")

            # Save raw data
            df.to_csv(self.raw_data_path, index=False, header=True)
            logging.info("Raw data saved at artifacts/raw.csv")

            # Train-Test Split
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train & test
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)
            logging.info("Train and test data saved successfully")

            logging.info("===== Data Ingestion Completed =====")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in Data Ingestion")
            raise CustomException(e, sys)
        

if __name__=="__main__":
    abc=DataIngestion()
    abc.initiate_data_ingestion()



