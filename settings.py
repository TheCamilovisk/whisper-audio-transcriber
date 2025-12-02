from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
    )

    bot_token: str
    output_dir: Path = ROOT / 'outputs'
    model_id: str = 'openai/whisper-small'
    use_cuda: bool = False

    @field_validator('use_cuda', mode='before')
    @classmethod
    def validate_use_cuda(cls, v):
        if isinstance(v, str):
            return v.lower() in {'true', '1', 'yes'}
        if isinstance(v, int):
            return bool(v)
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()
