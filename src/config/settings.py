"""
Application configuration using Pydantic settings.
"""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Letta Configuration
    letta_server_url: str = "http://localhost:8283"
    letta_api_key: str | None = None

    # Memory Backend
    memory_backend: Literal["letta", "llamaindex", "praxos"] = "letta"

    # LLM Configuration
    openai_api_key: str
    anthropic_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hetairos"
    postgres_user: str = "hetairos_user"
    postgres_password: str

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "hetairos"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Vector Store
    chroma_persist_dir: str = "./chroma_data"

    # Gmail Configuration
    gmail_client_id: str | None = None
    gmail_client_secret: str | None = None
    gmail_redirect_uri: str | None = None

    # Outlook Configuration
    outlook_client_id: str | None = None
    outlook_client_secret: str | None = None
    outlook_redirect_uri: str | None = None
    outlook_tenant_id: str = "common"

    # Azure Service Bus
    azure_service_bus_connection_string: str | None = None
    azure_service_bus_queue_name: str = "hetairos-events"

    # Worker Configuration
    consolidator_interval_seconds: int = 300  # 5 minutes
    email_indexer_batch_size: int = 100

    # Performance
    max_concurrent_agents: int = 10
    agent_timeout_seconds: int = 30

    @property
    def postgres_uri(self) -> str:
        """Construct PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_postgres_uri(self) -> str:
        """Construct async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Global settings instance
settings = Settings()
