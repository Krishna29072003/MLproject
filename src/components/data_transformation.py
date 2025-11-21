'''import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
import pickle


class DataTransformation:

    def __init__(self):
        logging.info("Data Transformation object created")

    def get_preprocessor(self, df):
        """
        Identify numerical and categorical columns 
        and return a ColumnTransformer (preprocessor)
        """
        num_cols = ['reading_score', 'writing_score']
        cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

        logging.info(f"Numerical columns: {num_cols}")
        logging.info(f"Categorical columns: {cat_cols}")

        # Numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ]
        )

        return preprocessor

    def start_transformation(self, train_path, test_path):

        try:
            logging.info("Reading train and test CSV files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Building preprocessor object")
            preprocessor = self.get_preprocessor(train_df)

            target_col = "math_score"     # Adjust if your target column name is different

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)

            logging.info("Transforming test data")
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            #os.makedirs("artifacts", exist_ok=True)
            #with open("artifacts/preprocessor.pkl", "wb") as f:
             #   pickle.dump(preprocessor, f)
            save_object("artifacts/preprocessor.pkl", preprocessor)
            logging.info("Preprocessor saved successfully as preprocessor.pkl")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                "artifacts/preprocessor.pkl"
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.start_transformation("artifacts/train.csv", "artifacts/test.csv")'''


import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


class DataTransformation:

    def __init__(self):
        logging.info("Data Transformation object created")

    def get_preprocessor(self):
        """
        Creates and returns a ColumnTransformer 
        containing numerical + categorical pipelines.
        """

        num_cols = ['reading_score', 'writing_score']
        cat_cols = [
            'gender',
            'race_ethnicity',
            'parental_level_of_education',
            'lunch',
            'test_preparation_course'
        ]

        logging.info(f"Numerical cols: {num_cols}")
        logging.info(f"Categorical cols: {cat_cols}")

        # Numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Combine both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads CSV → transforms → returns train_arr, test_arr, preprocessor_path
        """

        try:
            logging.info("Reading train and test CSV files")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Creating preprocessor object")
            preprocessor = self.get_preprocessor()

            target_col = "math_score"

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)

            logging.info("Transforming test data")
            X_test_transformed = preprocessor.transform(X_test)

            # Combine X and y for ModelTrainer
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            # Save preprocessor
            save_object("artifacts/preprocessor.pkl", preprocessor)
            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            # Return exactly what ModelTrainer needs
            return train_arr, test_arr, "artifacts/preprocessor.pkl"

        except Exception as e:
            raise CustomException(e, sys)




