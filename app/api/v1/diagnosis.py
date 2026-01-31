import logging
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.core import settings
from app.utils import image_processor
from app.schemas import PneumoniaPredictionResponse

api_router = APIRouter(prefix=settings.api.v1.lungcheck.prefix)
logger = logging.getLogger(__name__)

@api_router.post("/predict", response_model=PneumoniaPredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 1. Если не изображение - выдать исключение
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    try:
        # 2. Читаем изображение
        content = await file.read()

        # 3. Превращаем байты в тензор
        input_tensor = image_processor.process_image(content)

        # DEBUG: тензор создался правильно (1, 3, 224, 224)
        logger.info("Tensor successfully created. Shape: %s", input_tensor.shape)

        # 4. Заглушку по схеме
        return PneumoniaPredictionResponse(
            filename=file.filename,
            prediction="NORMAL",
            confidence=0.0,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )
    finally:
        await file.close()