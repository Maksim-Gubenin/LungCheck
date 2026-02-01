import logging
from datetime import datetime

import torch
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import db_helper, settings
from app.core.models import Prediction
from app.schemas import PneumoniaPredictionResponse
from app.utils import image_processor

api_router = APIRouter(prefix=settings.api.v1.lungcheck.prefix)
logger = logging.getLogger(__name__)


@api_router.post("/predict", response_model=PneumoniaPredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(db_helper.session_getter),
) -> PneumoniaPredictionResponse | None:
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

        new_prediction = Prediction(
            filename=file.filename,
            prediction=prediction_label,
            confidence=float(confidence.item()),
        )

        session.add(new_prediction)
        await session.commit()
        await session.refresh(new_prediction)

        return PneumoniaPredictionResponse(
            filename=new_prediction.filename,
            prediction=new_prediction.prediction,
            confidence=round(new_prediction.confidence, 4),
            timestamp=new_prediction.created_at,
        )

    except Exception as e:
        logger.exception("Error during prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}",
        ) from e
    finally:
        await file.close()


@api_router.get("/history", response_model=list[PneumoniaPredictionResponse])
async def get_history(
    session: AsyncSession = Depends(db_helper.session_getter), limit: int = 10
) -> list[PneumoniaPredictionResponse]:
    stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    predictions = result.scalars().all()

    return [
        PneumoniaPredictionResponse(
            filename=p.filename,
            prediction=p.prediction,
            confidence=p.confidence,
            timestamp=p.created_at,
        )
        for p in predictions
    ]
