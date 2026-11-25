from __future__ import annotations

import shlex
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.embeddings import Embeddings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Proveedor LLM: "agy" | "databricks" | "azure_openai" | "gemini" | "openai"
    llm_provider: str = "agy"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0

    # Configuración Antigravity CLI (`agy`) para desarrollo local
    agy_command: str = "agy"
    agy_subcommand: str = ""
    agy_extra_args: str = ""
    agy_input_mode: str = "stdin"
    agy_delay_seconds: float = 0.5

    # API Keys (solo si usas proveedores cloud por API)
    google_api_key: str = ""
    openai_api_key: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"

    # Embeddings: "local" | "databricks" | "gemini" | "openai"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Backend RAG: "local_hybrid" | "azure_search"
    retriever_backend: str = "local_hybrid"
    chroma_persist_dir: str = "data/outputs/chroma_db"
    azure_search_endpoint: str = ""
    azure_search_index: str = "easa-part-is"
    azure_search_key: str = ""

    # Parámetros del agente
    max_reflection_loops: int = 2
    confidence_threshold: float = 0.75
    top_k_clauses: int = 5


settings = Settings()


def get_llm(temperature: float | None = None) -> Any:
    temp = settings.llm_temperature if temperature is None else temperature

    if settings.llm_provider == "agy":
        from src.llm_agy import AgyCLILLM
        return AgyCLILLM(
            command=settings.agy_command,
            subcommand=settings.agy_subcommand,
            extra_args=shlex.split(settings.agy_extra_args) if settings.agy_extra_args else [],
            input_mode=settings.agy_input_mode,
            delay_seconds=settings.agy_delay_seconds,
        )

    if settings.llm_provider == "databricks":
        from databricks_langchain import ChatDatabricks  # type: ignore[import-not-found]
        return ChatDatabricks(endpoint=settings.llm_model, temperature=temp)

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=temp,
            google_api_key=settings.google_api_key or None,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.openai_api_key or None,
        )

    if settings.llm_provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.llm_model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            temperature=temp,
        )

    raise ValueError(f"LLM_PROVIDER no soportado: {settings.llm_provider}")


def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if settings.embedding_provider == "databricks":
        from databricks_langchain import DatabricksEmbeddings  # type: ignore[import-not-found]
        return DatabricksEmbeddings(endpoint=settings.embedding_model)

    if settings.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key or None,
        )

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
        )

    raise ValueError(f"EMBEDDING_PROVIDER no soportado: {settings.embedding_provider}")