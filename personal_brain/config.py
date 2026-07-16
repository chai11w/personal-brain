from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatModelConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    enabled: bool = False
    timeout_seconds: int = 60


@dataclass(frozen=True)
class EmbeddingModelConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-small"
    enabled: bool = False
    timeout_seconds: int = 60
    dimension: int | None = None


@dataclass(frozen=True)
class BrainConfig:
    database_path: Path
    memory_dir: Path
    brain_index_path: Path
    default_source: str = "cli"
    chat_model: ChatModelConfig = ChatModelConfig()
    embedding_model: EmbeddingModelConfig = EmbeddingModelConfig()


def load_config(path: str | Path = "config.json") -> BrainConfig:
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    chat_data = data.get("chat_model", data.get("llm", {}))
    chat_model = ChatModelConfig(
        provider=chat_data.get("provider", "openai_compatible"),
        base_url=chat_data.get("base_url", "https://api.openai.com/v1"),
        api_key_env=chat_data.get("api_key_env", "OPENAI_API_KEY"),
        model=chat_data.get("model", "gpt-4.1-mini"),
        enabled=bool(chat_data.get("enabled", False)),
        timeout_seconds=int(chat_data.get("timeout_seconds", 60)),
    )
    embedding_data = data.get("embedding_model", data.get("embedding", {}))
    dimension = embedding_data.get("dimension")
    embedding_model = EmbeddingModelConfig(
        provider=embedding_data.get("provider", "openai_compatible"),
        base_url=embedding_data.get("base_url", "https://api.openai.com/v1"),
        api_key_env=embedding_data.get("api_key_env", "OPENAI_API_KEY"),
        model=embedding_data.get("model", "text-embedding-3-small"),
        enabled=bool(embedding_data.get("enabled", False)),
        timeout_seconds=int(embedding_data.get("timeout_seconds", 60)),
        dimension=int(dimension) if dimension is not None else None,
    )

    return BrainConfig(
        database_path=Path(data.get("database_path", "data/personal_brain.sqlite3")),
        memory_dir=Path(data.get("memory_dir", "memory")),
        brain_index_path=Path(data.get("brain_index_path", "brain_index.json")),
        default_source=data.get("default_source", "cli"),
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
