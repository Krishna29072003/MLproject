import pickle
import os
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5

        r2 = r2_score(y_true, y_pred)

        return mae, rmse, r2

    except Exception as e:
        raise CustomException(e, sys)
