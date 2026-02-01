from datetime import datetime

from pydantic import BaseModel


class PneumoniaPredictionResponse(BaseModel):
    """
    Data Transfer Object (DTO) for pneumonia diagnosis results.

    This schema defines the structure of the API response, providing
    the user with the final classification result and the model's confidence.

    Attributes:
        filename: Name of the processed image file.
        prediction: Predicted class label ('NORMAL' or 'PNEUMONIA').
        confidence: Model confidence score as a percentage (0.0 to 100.0).
        timestamp: The exact time when the diagnosis was generated.
    """
    filename: str
    prediction: str
    confidence: float
    timestamp: datetime
