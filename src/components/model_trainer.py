import os
import sys
from dataclasses import dataclass

from sklearn.metrics import f1_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from src.constants import MODELS, GRID_PARAMS


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = MODELS
            grid_params = GRID_PARAMS

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, grid_params=grid_params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Found best model on both training and testing dataset: {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            best_model_score = f1_score(y_test, predicted)
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
