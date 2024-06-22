import os
import sys
from fastapi import FastAPI
from starlette import status
from typing import List
from src.pipeline.predict_request import PredictRequest
from src.utils import load_object
from src.exception import CustomException

app = FastAPI()


@app.get('/', status_code=status.HTTP_200_OK)
async def get_status():
    return {'status': 'OK'}


@app.post('/predict', status_code=status.HTTP_200_OK)
async def predict(predict_request: PredictRequest):
    features = [[
        predict_request.production_volume,
        predict_request.production_cost,
        predict_request.supplier_quantity,
        predict_request.delivery_delay,
        predict_request.defect_rate,
        predict_request.quality_score,
        predict_request.maintenance_hours,
        predict_request.downtime_percentage,
        predict_request.inventory_turnover,
        predict_request.stock_out_rate,
        predict_request.worker_productivity,
        predict_request.safety_incidents,
        predict_request.energy_consumption,
        predict_request.energy_efficiency,
        predict_request.additive_process_time,
        predict_request.additive_material_cost
    ]]

    prediction = predict_defect_level(features)[0]
    defect_level = 'Low Defect' if prediction == 0 else 'High Defect'
    return {
        'prediction': defect_level
    }


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
