"""LLMClient dual mode — Ollama local (opensource) + OpenAI fallback.

Este modulo provee una interfaz unificada para generar completions con dos
backends intercambiables:

    - ollama: modelo local (Llama 3.1 8B Q4) corriendo en la 3060 remota
      o en la M2 como fallback. Satisface el requisito opensource del reto.
    - openai: API remota (gpt-4o-mini por default) como fallback de red.

Modo "auto" hace un probe HTTP a /api/tags y cae a OpenAI si Ollama no
responde en <2 segundos.

Variables de entorno (ver .env.example):
    LLM_BACKEND=auto             # "ollama" | "openai" | "auto"
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente tecnico del equipo SUNASS en el Datathon 2026. "
    "Respondes en espanol formal, preciso y sin adornos. "
    "Cuando se te da contexto recuperado (RAG), citas explicitamente los "
    "archivos y secciones que usaste. Si la respuesta no esta en el "
    "contexto, dices 'No tengo esa informacion en el corpus' sin inventar."
)


@dataclass(frozen=True)
class LLMMessage:
    """Un turno de la conversacion."""

    role: Role
    content: str


@dataclass(frozen=True)
class LLMConfig:
    """Configuracion inmutable del cliente."""

    backend: Literal["ollama", "openai"]
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 120.0


class LLMClient:
    """Cliente LLM unificado. Despacha a Ollama u OpenAI segun config."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._ollama_client: Any = None
        self._openai_client: Any = None
        logger.info(
            "LLMClient: backend=%s model=%s base_url=%s",
            config.backend,
            config.model,
            config.base_url or "(n/a)",
        )

    def chat_stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        """Genera tokens del modelo a medida que llegan.

        Args:
            messages: turnos de la conversacion (system/user/assistant).

        Yields:
            Fragmentos de texto; concatenados forman la respuesta completa.
        """
        if self.config.backend == "ollama":
            yield from self._ollama_stream(messages)
        elif self.config.backend == "openai":
            yield from self._openai_stream(messages)
        else:
            raise ValueError(f"Backend desconocido: {self.config.backend!r}")

    def chat(self, messages: list[LLMMessage]) -> str:
        """Version no-stream: agrupa todos los tokens en un solo string."""
        return "".join(self.chat_stream(messages))

    # ------------------------------------------------------------------ Ollama

    def _ollama_stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        import ollama  # import tardio para no pagar el costo si no se usa

        if self._ollama_client is None:
            self._ollama_client = ollama.Client(
                host=self.config.base_url, timeout=self.config.timeout
            )
        stream = self._ollama_client.chat(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        for chunk in stream:
            # ollama client devuelve dict-like; acceso seguro
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            if msg is None:
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if content:
                yield content

    # ------------------------------------------------------------------ OpenAI

    def _openai_stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        from openai import OpenAI

        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.config.api_key, timeout=self.config.timeout)
        stream = self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


# --------------------------------------------------------------- Factory helpers


def probe_ollama(base_url: str, timeout: float = 2.0) -> bool:
    """Consulta /api/tags del servidor Ollama; True si responde 200."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=timeout)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
        logger.debug("Probe Ollama fallo en %s: %s", url, exc)
        return False
    return response.status_code == 200


def build_llm_client_from_env(env_file: str | None = None) -> LLMClient:
    """Construye LLMClient leyendo variables de entorno.

    Resolucion del backend:
        - LLM_BACKEND=ollama  -> fuerza Ollama; error si no responde al usar.
        - LLM_BACKEND=openai  -> fuerza OpenAI; error si OPENAI_API_KEY vacia.
        - LLM_BACKEND=auto    -> probe Ollama; si falla, usa OpenAI.

    Raises:
        RuntimeError: si el modo auto no encuentra ningun backend viable.
        ValueError: si LLM_BACKEND no es reconocido.
    """
    load_dotenv(dotenv_path=env_file)
    requested = os.getenv("LLM_BACKEND", "auto").lower()
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    openai_key = os.getenv("OPENAI_API_KEY") or None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    backend: str = requested
    if requested == "auto":
        if probe_ollama(ollama_url):
            backend = "ollama"
            logger.info("Modo auto: Ollama disponible en %s", ollama_url)
        elif openai_key:
            backend = "openai"
            logger.warning(
                "Modo auto: Ollama no responde en %s, cayendo a OpenAI (%s)",
                ollama_url,
                openai_model,
            )
        else:
            raise RuntimeError(
                f"Modo auto fallido: Ollama no responde en {ollama_url} y "
                "OPENAI_API_KEY no esta definida."
            )

    if backend == "ollama":
        return LLMClient(
            LLMConfig(backend="ollama", model=ollama_model, base_url=ollama_url)
        )
    if backend == "openai":
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY requerida pero no definida.")
        return LLMClient(
            LLMConfig(backend="openai", model=openai_model, api_key=openai_key)
        )
    raise ValueError(f"LLM_BACKEND invalido: {requested!r}")


def build_messages(
    user_query: str,
    *,
    system: str = DEFAULT_SYSTEM_PROMPT,
    rag_context: str | None = None,
    history: list[LLMMessage] | None = None,
) -> list[LLMMessage]:
    """Ensambla la secuencia de mensajes con system + RAG + historia + query.

    Args:
        user_query: pregunta del usuario (ultimo turno).
        system: prompt de sistema (por default, el del proyecto).
        rag_context: contexto recuperado del indice vectorial. Se inyecta como
            segundo mensaje de sistema con instruccion explicita de citar.
        history: turnos previos de la conversacion.

    Returns:
        Lista de LLMMessage lista para pasar a chat_stream().
    """
    messages: list[LLMMessage] = [LLMMessage(role="system", content=system)]
    if rag_context:
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "Contexto recuperado del corpus (cita los fragmentos que uses):\n\n"
                    f"{rag_context}\n\n"
                    "Si la respuesta no esta presente en el contexto, responde "
                    "'No tengo esa informacion en el corpus'."
                ),
            )
        )
    if history:
        messages.extend(history)
    messages.append(LLMMessage(role="user", content=user_query))
    return messages
