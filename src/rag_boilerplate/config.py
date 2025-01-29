"""Define core settings with Pydantic."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Store typed settings for Pydantic."""

    OPENAI_API_KEY: SecretStr = Field(
        ...,
        description="API key for OpenAI",
    )
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: SecretStr

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    """
    Return a Settings object.

    Returns
        Settings: Pydantic settings object

    """
    return Settings()
