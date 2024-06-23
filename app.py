from fastapi import FastAPI
from starlette import status
from src.pipeline.predict_request import PredictRequest
from src.pipeline.predict_pipeline import predict_defect_level

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
