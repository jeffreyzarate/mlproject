import os
import sys
from typing import List
from src.utils import load_object
from src.exception import CustomException


def predict_defect_level(features: List):
    try:
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        print("Before Loading")
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        print("After Loading")
        data_scaled = preprocessor.transform(features)
        prediction = model.predict(data_scaled)
        return prediction
    except Exception as e:
        raise CustomException(e, sys)
