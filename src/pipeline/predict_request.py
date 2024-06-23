from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    production_volume: int = Field()
    production_cost: float = Field()
    supplier_quantity: float = Field()
    delivery_delay: int = Field()
    defect_rate: float = Field()
    quality_score: float = Field()
    maintenance_hours: int = Field()
    downtime_percentage: float = Field()
    inventory_turnover: float = Field()
    stock_out_rate: float = Field()
    worker_productivity: float = Field()
    safety_incidents: int = Field()
    energy_consumption: float = Field()
    energy_efficiency: float = Field()
    additive_process_time: float = Field()
    additive_material_cost: float = Field()

    class Config:
        json_schema_extra = {
            "example": {
                "production_volume": 202,
                "production_cost": 13175.403783,
                "supplier_quantity": 86.648534,
                "delivery_delay": 1,
                "defect_rate": 3.121492,
                "quality_score": 63.463494,
                "maintenance_hours": 9,
                "downtime_percentage": 0.052343,
                "inventory_turnover": 8.630515,
                "stock_out_rate": 0.081322,
                "worker_productivity": 85.042379,
                "safety_incidents": 0,
                "energy_consumption": 2419.616785,
                "energy_efficiency": 0.468947,
                "additive_process_time": 5.551639,
                "additive_material_cost": 236.439301
            }
        }

