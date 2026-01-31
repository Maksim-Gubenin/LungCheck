from datetime import datetime

from pydantic import BaseModel


class PneumoniaPredictionResponse (BaseModel):
    filename: str
    prediction: str
    confidence: float
    timestamp: datetime
