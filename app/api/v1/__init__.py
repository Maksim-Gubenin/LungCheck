from fastapi import APIRouter

from app.api.v1.diagnosis import api_router as diagnosis_router
from app.core import settings

api_v1_router = APIRouter(prefix=settings.api.v1.prefix)
api_v1_router.include_router(diagnosis_router)
