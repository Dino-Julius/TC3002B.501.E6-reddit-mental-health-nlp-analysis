"""
Cliente local de Ollama y validación de salidas LLM para Fase 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from pydantic import BaseModel, Field, ValidationError, field_validator


class OllamaError(RuntimeError):
    """
    Representa errores de comunicación o contrato con Ollama.
    """


@dataclass(frozen=True)
class OllamaClient:
    """
    Cliente mínimo para la API local de Ollama.
    """

    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 120

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Ejecuta una petición POST JSON contra Ollama.
        """

        url = self.base_url.rstrip("/") + endpoint
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except error.URLError as exc:
            raise OllamaError(f"No se pudo conectar a Ollama en {url}: {exc}") from exc

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama devolvió una respuesta no JSON.") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("Ollama devolvió una respuesta con forma inesperada.")
        return parsed

    def embed(self, model: str, text: str) -> list[float]:
        """
        Genera un embedding para un texto usando el endpoint clásico de Ollama.
        """

        payload = {"model": model, "prompt": text}
        response = self._post_json("/api/embeddings", payload)
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise OllamaError("Ollama no devolvió un embedding válido.")
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise OllamaError("El embedding de Ollama contiene valores no numéricos.") from exc

    def generate_json(self, model: str, prompt: str) -> str:
        """
        Solicita una respuesta JSON no-streaming a un modelo generativo local.
        """

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        response = self._post_json("/api/generate", payload)
        output = response.get("response")
        if not isinstance(output, str) or not output.strip():
            raise OllamaError("Ollama no devolvió texto en la respuesta generativa.")
        return output


class LlmClassification(BaseModel):
    """
    Salida estructurada esperada del LLM para una publicación.
    """

    label: str
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    justification: str = ""

    @field_validator("label")
    @classmethod
    def normalizar_label(cls, value: str) -> str:
        """
        Acepta solo etiquetas binarias compatibles con el protocolo.
        """

        label = value.strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError("label debe ser 'yes' o 'no'.")
        return label

    @field_validator("evidence")
    @classmethod
    def limpiar_evidence(cls, value: list[str]) -> list[str]:
        """
        Normaliza evidencia textual breve sin conservar elementos vacíos.
        """

        return [str(item).strip()[:240] for item in value if str(item).strip()][:5]

    @field_validator("justification")
    @classmethod
    def limitar_justification(cls, value: str) -> str:
        """
        Limita la justificación para mantener artefactos compactos.
        """

        return value.strip()[:500]


def label_to_prediction(label: str) -> int:
    """
    Convierte yes/no a la codificación binaria del proyecto.
    """

    normalized = label.strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Etiqueta LLM no soportada: {label}")


def parse_llm_classification(raw_response: str) -> LlmClassification:
    """
    Parsea y valida una clasificación JSON generada por el LLM.
    """

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("La respuesta del LLM no es JSON válido.") from exc
    if not isinstance(payload, dict):
        raise ValueError("La respuesta del LLM debe ser un objeto JSON.")
    try:
        return LlmClassification.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("La respuesta del LLM no cumple el esquema esperado.") from exc


def construir_prompt_zero_shot(texto: str) -> str:
    """
    Construye el prompt de clasificación zero-shot para suicidalidad.
    """

    return f"""
You are evaluating an academic NLP benchmark for suicidal ideation detection.
Classify whether the Reddit post contains suicidal ideation or suicidal risk.

Return only valid JSON with this exact schema:
{{
  "label": "yes" or "no",
  "score": number from 0.0 to 1.0,
  "evidence": ["short quoted or paraphrased evidence"],
  "justification": "brief explanation"
}}

Rules:
- Use "yes" when the post expresses suicidal ideation, intent, planning,
  self-harm intent, or explicit desire not to live.
- Use "no" for depression, stress, sadness, or support-seeking without suicidal
  ideation or self-harm intent.
- The score is confidence for the "yes" class.
- Do not provide clinical advice. Do not add text outside the JSON.

Reddit post:
\"\"\"{texto[:6000]}\"\"\"
""".strip()


__all__ = [
    "LlmClassification",
    "OllamaClient",
    "OllamaError",
    "construir_prompt_zero_shot",
    "label_to_prediction",
    "parse_llm_classification",
]
