"""
Pruebas de contratos Ollama/LLM para Fase 3.
"""

from __future__ import annotations

import pytest

from reddit_mental_health.ollama import (
    construir_prompt_zero_shot,
    label_to_prediction,
    parse_llm_classification,
)


def test_parse_llm_classification_normaliza_respuesta_valida() -> None:
    """
    Verifica que una respuesta JSON válida sea normalizada.
    """

    result = parse_llm_classification(
        """
        {
          "label": "YES",
          "score": 0.87,
          "evidence": [" explicit intent to die "],
          "justification": "Contains suicidal ideation."
        }
        """
    )

    assert result.label == "yes"
    assert result.score == pytest.approx(0.87)
    assert result.evidence == ["explicit intent to die"]
    assert result.justification == "Contains suicidal ideation."


def test_parse_llm_classification_rechaza_json_invalido() -> None:
    """
    Verifica que el parser rechace salidas fuera del esquema.
    """

    with pytest.raises(ValueError, match="esquema esperado"):
        parse_llm_classification('{"label": "maybe", "score": 1.4}')


def test_parse_llm_classification_recorta_evidencia_larga() -> None:
    """
    Verifica que evidencia extensa del LLM no invalide una respuesta útil.
    """

    result = parse_llm_classification(
        """
        {
          "label": "yes",
          "score": 0.95,
          "evidence": ["a", "b", "c", "d", "e", "f"],
          "justification": "ok"
        }
        """
    )

    assert result.evidence == ["a", "b", "c", "d", "e"]


def test_label_to_prediction_convierte_yes_no() -> None:
    """
    Verifica conversión binaria de etiquetas LLM.
    """

    assert label_to_prediction("yes") == 1
    assert label_to_prediction(" NO ") == 0
    with pytest.raises(ValueError):
        label_to_prediction("unknown")


def test_construir_prompt_zero_shot_exige_json_y_no_diagnostico() -> None:
    """
    Verifica que el prompt preserve restricciones clave del reto.
    """

    prompt = construir_prompt_zero_shot("I do not want to keep living.")

    assert "Return only valid JSON" in prompt
    assert '"label": "yes" or "no"' in prompt
    assert "Do not provide clinical advice" in prompt
    assert "I do not want to keep living." in prompt
