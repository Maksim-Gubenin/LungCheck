import logging
from datetime import datetime

import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core import settings
from app.schemas import PneumoniaPredictionResponse
from app.utils import image_processor

api_router = APIRouter(prefix=settings.api.v1.lungcheck.prefix)
logger = logging.getLogger(__name__)


@api_router.post("/predict", response_model=PneumoniaPredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PneumoniaPredictionResponse | None:
    # 1. Если не изображение - выдать исключение
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image"
        )

    try:
        # 2. Читаем изображение
        content = await file.read()

        # 3. Превращаем байты в тензор
        input_tensor = image_processor.process_image(content)

        logger.info("Tensor successfully created. Shape: %s", input_tensor.shape)

        # >>>>>>>>>>><<<<<<<<<<<

        # Достаем модель из состояния приложения
        model = request.app.state.model

        # Выполняем анализ без вычисления градиентов
        with torch.no_grad():
            output = model(input_tensor)
            # Применяем Softmax для получения вероятностей
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Получаем вероятность и индекс класса с максимальной вероятностью
            confidence, predicted_class = torch.max(probabilities, 1)

        # Определяем название класса
        class_names = ["NORMAL", "PNEUMONIA"]
        prediction_label = class_names[int(predicted_class.item())]

        # 4. Возвращаем результат
        filename = file.filename or "unknown.jpg"
        return PneumoniaPredictionResponse(
            filename=filename,
            prediction=prediction_label,
            confidence=round(confidence.item(), 4),
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.exception("Error during prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}",
        ) from e
    finally:
        await file.close()
