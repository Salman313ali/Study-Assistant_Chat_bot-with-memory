import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    groq_api_key: Optional[str]
    hf_token: Optional[str]


def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
