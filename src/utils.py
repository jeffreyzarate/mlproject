import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import dill
import pickle

from src.exception import CustomException
from src.logger import logging


def initiate_train_test_split(df: pd.DataFrame, train_data_path: str, test_data_path: str):
    try:
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(train_data_path, index=False, header=True)

        test_set.to_csv(test_data_path, index=False, header=True)

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, grid_params: dict):
    try:
        report = {}

        for key, value in models.items():
            model = value
            params = grid_params.get(key)

            gs = GridSearchCV(model, params, cv=3, scoring='f1')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)

            report[key] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
