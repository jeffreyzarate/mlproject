import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dill

from src.exception import CustomException
from src.logger import logging


def initiate_train_test_split(df: pd.DataFrame, train_data_path: str, test_data_path: str):
    try:
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(train_data_path, index=False, header=True)

        test_set.to_csv(test_data_path, index=False, header=True)

    except Exception as e:
        raise CustomException(e, sys)


def save_processor(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


