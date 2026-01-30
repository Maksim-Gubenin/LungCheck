from fastapi import APIRouter

from app.api.v1 import api_v1_router
from app.core import settings

api_router = APIRouter(prefix=settings.api.prefix)
api_router.include_router(api_v1_router)
