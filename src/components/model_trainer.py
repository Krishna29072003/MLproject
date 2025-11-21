# src/components/model_trainer.py
import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test input/output features")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            best_model_name = None
            best_model_score = -999
            best_model = None

            logging.info("Training and evaluating models...")

            for name, model in models.items():
                logging.info(f"Training model: {name}")

                model.fit(X_train, y_train)

                y_pred_test = model.predict(X_test)

                _, _, r2 = evaluate_model(y_test, y_pred_test)

                logging.info(f"{name} R2 Score: {r2}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model_name = name
                    best_model = model

            logging.info(f"Best model selected: {best_model_name} with R2 score {best_model_score}")

            # If best accuracy is below threshold → fail
            if best_model_score < 0.6:
                raise CustomException("No good model found. Best R2 < 0.6", sys)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully.")

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
        


'''if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model_name, best_model_score= trainer.initiate_model_training()'''



