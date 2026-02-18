import pytest
import pytest_asyncio
import torch
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.main import app
from app.core import db_helper
from app.core.models.base import Base

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    """Создаем таблицы один раз для всей сессии."""
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def session():
    """Чистая сессия для каждого теста."""
    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest_asyncio.fixture
async def client(session):
    """Клиент с моками ML и БД."""
    app.dependency_overrides[db_helper.session_getter] = lambda: session

    with patch("app.core.ml.model_loader.model_loader.load_model") as mock_load:
        mock_resnet = MagicMock()
        mock_resnet.return_value = torch.tensor([[-1.0, 5.0]])
        mock_load.return_value = mock_resnet

        with patch("app.utils.image_processor.process_image") as mock_proc:
            mock_proc.return_value = torch.randn(1, 3, 224, 224)

            async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test"
            ) as ac:
                yield ac

    app.dependency_overrides.clear()