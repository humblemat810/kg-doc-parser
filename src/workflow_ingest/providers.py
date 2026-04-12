"""Provider-neutral OCR, parser, and embedding adapters.

This module keeps vendor-specific imports behind small factory functions so the
workflow code can stay neutral. The concrete vendor is selected by config, not
by the caller.

Quick examples
--------------
- OCR with Google GenAI:
  - KG_DOC_OCR_PROVIDER=gemini
  - KG_DOC_OCR_MODEL=gemini-2.5-flash

- OCR with a local Ollama vision model:
  - KG_DOC_OCR_PROVIDER=ollama
  - KG_DOC_OCR_MODEL=llava:latest
  - KG_DOC_OCR_BASE_URL=http://127.0.0.1:11434

- Parser/LLM with OpenAI Chat Completions:
  - KG_DOC_PARSER_PROVIDER=openai
  - KG_DOC_PARSER_MODEL=gpt-4.1-mini
  - KG_DOC_PARSER_API_KEY_ENV=OPENAI_API_KEY

- Parser/LLM with Google Vertex AI:
  - KG_DOC_PARSER_PROVIDER=vertex
  - KG_DOC_PARSER_MODEL=gemini-2.5-pro
  - KG_DOC_PARSER_PROJECT=my-project
  - KG_DOC_PARSER_LOCATION=us-central1

- Parser/LLM with Ollama:
  - KG_DOC_PARSER_PROVIDER=ollama
  - KG_DOC_PARSER_MODEL=llama3.1
  - KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434

- Embeddings with a fake deterministic function for CI:
  - KG_DOC_EMBED_PROVIDER=fake
  - KG_DOC_EMBED_MODEL=kg-doc-parser-workflow-embedding-v1

Cookbook example
----------------
If you are parsing a cooking recipe, you can keep OCR on Gemini but route the
parser to OpenAI or Ollama:

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="gemini", model="gemini-2.5-flash"),
        parser=ProviderEndpointConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
        ),
    )

That means the OCR step extracts the page text, and the parser step can then
turn the recipe into structured fields such as ingredients, tools, actions,
and inferred sections without changing workflow orchestration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated, Any, Callable, ClassVar, Literal, Optional, Protocol, Union, runtime_checkable, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from pydantic_extension.model_slicing import BackendField, FrontendField
from pydantic_extension.model_slicing.mixin import DtoField, ExcludeMode, LLMField, ModeSlicingMixin


class _FakeStructuredResponse:
    def __init__(self, schema, payload: dict[str, Any]):
        self.schema = schema
        self.payload = payload

    def invoke(self, messages, config=None):
        parsed = self.schema.model_validate(self.payload)
        return {"parsed": parsed, "raw": None, "parsing_error": None}


class FakeChatModel:
    """Minimal structured-output compatible chat model for tests."""

    def __init__(self, *, payload_factory: Callable[[Any], dict[str, Any]] | None = None) -> None:
        self.payload_factory = payload_factory or _default_schema_payload

    def with_structured_output(self, schema, include_raw: bool = True):
        payload = self.payload_factory(schema)
        return _FakeStructuredResponse(schema, payload)


def _default_schema_payload(schema) -> dict[str, Any]:
    def _value_for_field(field) -> Any:
        annotation = getattr(field, "annotation", None)
        origin = get_origin(annotation)
        args = get_args(annotation)
        if annotation is str:
            return ""
        if annotation is bool:
            return False
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if origin is list or annotation is list:
            return []
        if origin is dict or annotation is dict:
            return {}
        if origin is tuple:
            return []
        if origin is Literal and args:
            return args[0]
        if origin is Union and type(None) in args:
            return None
        if hasattr(annotation, "model_fields"):
            return _default_schema_payload(annotation)
        default = getattr(field, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            return default
        return None

    payload: dict[str, Any] = {}
    for name, field in getattr(schema, "model_fields", {}).items():
        value = _value_for_field(field)
        if value is not None:
            payload[name] = value
    return payload


@runtime_checkable
class ChatModelProvider(Protocol):
    def build(self, *, callbacks: list[Any] | None = None) -> Any: ...


@runtime_checkable
class EmbeddingFunctionProvider(Protocol):
    def build(self) -> Callable[[list[str]], list[list[float]]]: ...


class ProviderEndpointConfig(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    provider: Annotated[
        Literal["gemini", "ollama", "openai", "vertex", "fake"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "gemini"
    model: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()] = "gemini-2.5-flash"
    temperature: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()] = 0.1
    base_url: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    api_key_env: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    project: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    location: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    max_retries: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 2


class EmbeddingProviderConfig(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    provider: Annotated[
        Literal["fake", "openai", "vertex", "ollama"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "fake"
    model: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()] = "kg-doc-parser-workflow-embedding-v1"
    dimension: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 2
    base_url: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    api_key_env: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None


class WorkflowProviderSettings(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    ocr: Annotated[ProviderEndpointConfig, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=ProviderEndpointConfig
    )
    parser: Annotated[ProviderEndpointConfig, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=ProviderEndpointConfig
    )
    embedding: Annotated[EmbeddingProviderConfig, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=EmbeddingProviderConfig
    )

    @classmethod
    def from_env(cls) -> "WorkflowProviderSettings":
        def _env(name: str, default: str | None = None) -> str | None:
            value = os.getenv(name)
            return value if value not in {None, ""} else default

        return cls(
            ocr=ProviderEndpointConfig(
                provider=str(_env("KG_DOC_OCR_PROVIDER", "gemini")),
                model=str(_env("KG_DOC_OCR_MODEL", "gemini-2.5-flash")),
                temperature=float(_env("KG_DOC_OCR_TEMPERATURE", "0.1")),
                base_url=_env("KG_DOC_OCR_BASE_URL"),
                api_key_env=_env("KG_DOC_OCR_API_KEY_ENV"),
                project=_env("KG_DOC_OCR_PROJECT"),
                location=_env("KG_DOC_OCR_LOCATION"),
                max_retries=int(_env("KG_DOC_OCR_MAX_RETRIES", "2")),
            ),
            parser=ProviderEndpointConfig(
                provider=str(_env("KG_DOC_PARSER_PROVIDER", "gemini")),
                model=str(_env("KG_DOC_PARSER_MODEL", "gemini-2.5-flash")),
                temperature=float(_env("KG_DOC_PARSER_TEMPERATURE", "0.1")),
                base_url=_env("KG_DOC_PARSER_BASE_URL"),
                api_key_env=_env("KG_DOC_PARSER_API_KEY_ENV"),
                project=_env("KG_DOC_PARSER_PROJECT"),
                location=_env("KG_DOC_PARSER_LOCATION"),
                max_retries=int(_env("KG_DOC_PARSER_MAX_RETRIES", "2")),
            ),
            embedding=EmbeddingProviderConfig(
                provider=str(_env("KG_DOC_EMBED_PROVIDER", "fake")),
                model=str(_env("KG_DOC_EMBED_MODEL", "kg-doc-parser-workflow-embedding-v1")),
                dimension=int(_env("KG_DOC_EMBED_DIMENSION", "2")),
                base_url=_env("KG_DOC_EMBED_BASE_URL"),
                api_key_env=_env("KG_DOC_EMBED_API_KEY_ENV"),
            ),
        )


def _embedding_vector(text: str, *, dimension: int) -> list[float]:
    checksum = sum(ord(ch) for ch in text or "")
    return [
        float((len(text) + idx + 1) % 97 + 1 + (checksum % 13))
        for idx in range(max(1, dimension))
    ]


@dataclass
class _CallableEmbeddingFunction:
    name_value: str
    dimension: int
    provider: str

    def name(self) -> str:
        return self.name_value

    def __call__(self, input):
        vectors = []
        for value in input:
            vectors.append(_embedding_vector(str(value or ""), dimension=self.dimension))
        return vectors


def build_embedding_function(
    spec: EmbeddingProviderConfig | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """Build a configurable embedding callable.

    Supported providers currently include fake, OpenAI, Vertex AI, and Ollama.
    The fake provider is deterministic and preferred for unit tests.
    """
    spec = spec or EmbeddingProviderConfig()
    if spec.provider == "fake":
        return _CallableEmbeddingFunction(
            name_value=spec.model,
            dimension=spec.dimension,
            provider=spec.provider,
        )

    def _build_langchain_embeddings() -> Any:
        if spec.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {"model": spec.model}
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            if spec.api_key_env and os.getenv(spec.api_key_env):
                kwargs["api_key"] = os.getenv(spec.api_key_env)
            return OpenAIEmbeddings(**kwargs)
        if spec.provider == "vertex":
            from langchain_google_vertexai import VertexAIEmbeddings

            kwargs = {"model_name": spec.model}
            if spec.project:
                kwargs["project"] = spec.project
            if spec.location:
                kwargs["location"] = spec.location
            return VertexAIEmbeddings(**kwargs)
        if spec.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            kwargs = {"model": spec.model}
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            return OllamaEmbeddings(**kwargs)
        raise ValueError(f"unsupported embedding provider: {spec.provider}")

    embeddings = _build_langchain_embeddings()

    class _LangChainEmbeddingFunction:
        def name(self) -> str:
            return spec.model

        def __call__(self, input):
            texts = [str(value or "") for value in input]
            if hasattr(embeddings, "embed_documents"):
                return embeddings.embed_documents(texts)
            if hasattr(embeddings, "embed_query"):
                return [embeddings.embed_query(text) for text in texts]
            raise TypeError(f"unsupported embedding backend: {type(embeddings)!r}")

    return _LangChainEmbeddingFunction()


def build_chat_model(
    spec: ProviderEndpointConfig | None = None,
    *,
    callbacks: list[Any] | None = None,
):
    """Build a vendor-specific chat model behind a stable adapter boundary.

    Supported providers currently include gemini, openai, ollama, vertex, and
    fake. Callers should choose the provider via config and treat the returned
    object as a LangChain-compatible chat model.
    """
    spec = spec or ProviderEndpointConfig()
    callbacks = callbacks or []
    if spec.provider == "fake":
        return FakeChatModel()
    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, Any] = {"model": spec.model, "temperature": spec.temperature, "callbacks": callbacks}
        if spec.api_key_env and os.getenv(spec.api_key_env):
            kwargs["google_api_key"] = os.getenv(spec.api_key_env)
        return ChatGoogleGenerativeAI(**kwargs)
    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": spec.model, "temperature": spec.temperature, "callbacks": callbacks}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if spec.api_key_env and os.getenv(spec.api_key_env):
            kwargs["api_key"] = os.getenv(spec.api_key_env)
        return ChatOpenAI(**kwargs)
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": spec.model, "temperature": spec.temperature, "callbacks": callbacks}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        return ChatOllama(**kwargs)
    if spec.provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI

        kwargs = {"model": spec.model, "temperature": spec.temperature, "callbacks": callbacks}
        if spec.project:
            kwargs["project"] = spec.project
        if spec.location:
            kwargs["location"] = spec.location
        return ChatVertexAI(**kwargs)
    raise ValueError(f"unsupported chat provider: {spec.provider}")


def build_chat_model_for_role(
    role: Literal["ocr", "parser"],
    spec: WorkflowProviderSettings | None = None,
    *,
    callbacks: list[Any] | None = None,
):
    """Build the chat model used for either OCR or parsing.

    Examples:
    - role="ocr" with KG_DOC_OCR_PROVIDER=gemini for image OCR.
    - role="parser" with KG_DOC_PARSER_PROVIDER=openai for recipe extraction.
    - role="parser" with KG_DOC_PARSER_PROVIDER=ollama for local models.
    """
    settings = spec or WorkflowProviderSettings.from_env()
    chat_spec = settings.ocr if role == "ocr" else settings.parser
    return build_chat_model(chat_spec, callbacks=callbacks)
