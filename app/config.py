"""
Configuration settings for CodeBlitz Microservice.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Application
    # ═══════════════════════════════════════════════════════════════
    APP_NAME: str = "CodeBlitz Microservice"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    
    # ═══════════════════════════════════════════════════════════════
    # PISTON API Configuration
    # ═══════════════════════════════════════════════════════════════
    PISTON_URL: str = "http://localhost:2000"
    PISTON_EXECUTE_TIMEOUT: int = 30  # seconds per execution
    PISTON_MAX_RETRIES: int = 3
    PISTON_RETRY_DELAY: float = 1.0  # seconds between retries
    PISTON_MAX_CONCURRENT: int = 5  # max parallel executions
    
    # Language versions supported (Python, Java, C++ only)
    PISTON_PYTHON_VERSION: str = "3.10.0"
    PISTON_JAVA_VERSION: str = "15.0.2"
    PISTON_CPP_VERSION: str = "10.2.0"
    
    # ═══════════════════════════════════════════════════════════════
    # Supabase Configuration
    # ═══════════════════════════════════════════════════════════════
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""  # anon/service role key
    
    # ═══════════════════════════════════════════════════════════════
    # OpenAI Configuration (regular API - for production)
    # ═══════════════════════════════════════════════════════════════
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.2"
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TEMPERATURE: float = 0.7
    
    # ═══════════════════════════════════════════════════════════════
    # Azure AI Foundry Configuration (for testing with best models)
    # ═══════════════════════════════════════════════════════════════
    AZURE_AI_FOUNDRY_ENDPOINT: str = "https://astryx-ai-providers-resource.openai.azure.com"
    AZURE_AI_FOUNDRY_API_KEY: str = ""
    USE_AZURE_AI_FOUNDRY: bool = True  # Set to False to use regular OpenAI
    
    # ═══════════════════════════════════════════════════════════════
    # Generation Settings
    # ═══════════════════════════════════════════════════════════════
    MAX_GENERATION_RETRIES: int = 3
    MAX_VALIDATION_RETRIES: int = 2
    SIMILARITY_THRESHOLD: float = 0.92  # for deduplication
    
    # ═══════════════════════════════════════════════════════════════
    # Supported Languages (Python, Java, C++ only)
    # ═══════════════════════════════════════════════════════════════
    @property
    def SUPPORTED_LANGUAGES(self) -> dict:
        return {
            "python": {
                "version": self.PISTON_PYTHON_VERSION,
                "filename": "main.py"
            },
            "java": {
                "version": self.PISTON_JAVA_VERSION,
                "filename": "Main.java"
            },
            "cpp": {
                "language": "c++",  # PISTON uses "c++" not "cpp"
                "version": self.PISTON_CPP_VERSION,
                "filename": "main.cpp"
            }
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
