from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from typing import Any, Type, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).strip()


def _extract_json_block(text: str) -> str:
    """Extrae el primer objeto JSON válido incluso si la CLI imprime texto o fences alrededor."""
    cleaned = _strip_ansi(text)

    # 1. Bloque ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # 2. Primer '{' hasta último '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()

    return cleaned


def _messages_to_prompt(messages: list[BaseMessage] | str) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for msg in messages:
        role = msg.type.upper()
        parts.append(f"[{role}]\n{msg.content}")
    return "\n\n".join(parts)


class AgyStructuredRunnable:
    """Emula llm.with_structured_output(Schema) delegando en `agy ask`."""

    def __init__(self, llm: AgyCLILLM, schema: Type[T], max_retries: int = 2) -> None:
        self.llm = llm
        self.schema = schema
        self.max_retries = max_retries

    def invoke(self, messages: list[BaseMessage] | str) -> T:
        base_prompt = _messages_to_prompt(messages)
        json_schema = json.dumps(self.schema.model_json_schema(), ensure_ascii=False, indent=2)

        full_prompt = (
            f"{base_prompt}\n\n"
            "INSTRUCCIÓN DE FORMATO OBLIGATORIA:\n"
            "Responde ÚNICAMENTE con un objeto JSON válido que cumpla exactamente este JSON Schema.\n"
            "No añadas explicaciones, saludos ni texto fuera del JSON.\n\n"
            f"JSON SCHEMA:\n{json_schema}"
        )

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            raw_output = self.llm.run_cli(full_prompt)
            candidate = _extract_json_block(raw_output)
            try:
                return self.schema.model_validate_json(candidate)
            except Exception as exc:
                last_err = exc
                full_prompt = (
                    f"{full_prompt}\n\n"
                    f"ERROR EN TU RESPUESTA ANTERIOR (intento {attempt}): {exc}\n"
                    "Corrige el JSON y devuelve SOLO el objeto JSON válido."
                )
                time.sleep(self.llm.delay_seconds)

        raise ValueError(
            f"agy ask no devolvió JSON válido para {self.schema.__name__} tras {self.max_retries} intentos: {last_err}"
        )


class AgyCLILLM:
    """Cliente ligero que invoca `agy ask` por terminal para evitar saturación de API keys."""

    def __init__(
        self,
        command: str = "agy",
        subcommand: str = "ask",
        extra_args: list[str] | None = None,
        input_mode: str = "arg",  # "arg" (pasa prompt como argumento) o "stdin" (por entrada estándar)
        timeout_seconds: int = 120,
        delay_seconds: float = 0.5,
    ) -> None:
        self.command = command
        self.subcommand = subcommand
        self.extra_args = extra_args or []
        self.input_mode = input_mode
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds

    def _resolve_executable(self) -> str:
        exe = shutil.which(self.command)
        if not exe:
            raise FileNotFoundError(
                f"No se encontró '{self.command}' en el PATH. Verifica que `agy ask` funciona en la terminal de VS Code."
            )
        return exe

    def run_cli(self, prompt: str) -> str:
        exe = self._resolve_executable()
        cmd = [exe]
        if self.subcommand:
            cmd.append(self.subcommand)
        cmd.extend(self.extra_args)

        stdin_payload: str | None = None
        if self.input_mode == "stdin":
            stdin_payload = prompt
        else:
            cmd.append(prompt)

        proc = subprocess.run(
            cmd,
            input=stdin_payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            shell=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"`{self.command} {self.subcommand}` falló (code {proc.returncode}): {proc.stderr.strip()}"
            )

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        return _strip_ansi(proc.stdout)

    def invoke(self, messages: list[BaseMessage] | str) -> str:
        return self.run_cli(_messages_to_prompt(messages))

    def with_structured_output(self, schema: Type[T]) -> AgyStructuredRunnable:
        return AgyStructuredRunnable(self, schema)