from pathlib import Path

from pydantic import BaseModel, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_TEMPLATE_PATH = PROJECT_ROOT / ".env.template"


class DatabaseConfig(BaseModel):
    """
    Database connection settings and SQLAlchemy engine configuration.

    Attributes:
        url: DSN for PostgreSQL connection.
        echo: Enable SQLAlchemy logging.
        pool_size: Number of connections to keep in the pool.
        naming_convention: Standardized names for constraints (Alembic friendly).
    """
    url: PostgresDsn
    echo: bool = False
    echo_pool: bool = False
    pool_size: int = 50
    max_overflow: int = 10

    naming_convention: dict[str, str] = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }


class RunConfig(BaseModel):
    """
    Server runtime configuration.

    Attributes:
        host: The network interface to bind the server to.
        port: The TCP port on which the application will listen.
    """
    host: str = "0.0.0.0"
    port: int = 8000


class LungCheckPrefix(BaseModel):
    """
    Sub-prefix configuration for the LungCheck diagnostic module.
    """
    prefix: str = "/lungcheck"


class MLConfig(BaseModel):
    """
    Machine Learning specific configuration.

    Attributes:
        model_path: Relative path to the trained .pth weights file.
    """
    model_path: str = "models/pneumonia_resnet18.pth"


class ApiV1Prefix(BaseModel):
    """
    Configuration for Version 1 of the API.

    Groups all v1-specific endpoints and their respective prefixes.
    """
    prefix: str = "/v1"
    lungcheck: LungCheckPrefix = LungCheckPrefix()


class ApiPrefix(BaseModel):
    """
    Global API routing configuration.

    Defines the base entry point for all API endpoints and
    manages version-based routing.
    """
    prefix: str = "/api"
    v1: ApiV1Prefix = ApiV1Prefix()


class Settings(BaseSettings):
    """
    Main settings class that orchestrates all sub-configs.

    Loads data from environment variables and .env files using
    the APP_CONFIG__ prefix and double-underscore delimiter for nested fields.
    """
    model_config = SettingsConfigDict(
        env_file=(ENV_TEMPLATE_PATH, ENV_PATH),
        case_sensitive=False,
        env_nested_delimiter="__",
        env_prefix="APP_CONFIG__",
    )
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    run: RunConfig = RunConfig()
    api: ApiPrefix = ApiPrefix()
    ml_config: MLConfig = MLConfig()
    db: DatabaseConfig


settings = Settings()
