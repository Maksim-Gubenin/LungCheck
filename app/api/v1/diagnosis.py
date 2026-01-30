from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.core import settings

api_router = APIRouter(prefix=settings.api.v1.lungcheck.prefix)

@api_router.post("/predict")
async def upload_image(files: Annotated[list[bytes], File()]):
    return {"file_sizes": [len(file) for file in files]}


@api_router.post("/uploadfiles")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}
