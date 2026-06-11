"""
Pruebas del flujo LLM de Fase 3.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.phase3_llm import (
    FewShotExample,
    clasificar_publicaciones_llm,
    ejemplos_few_shot_a_json,
    evaluar_y_guardar_llm,
    seleccionar_ejemplos_few_shot,
    serializar_evidence_para_csv,
)


class FakeGenerativeClient:
    """
    Cliente generativo con respuestas predefinidas.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def generate_json(self, model: str, prompt: str) -> str:
        del model
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_clasificar_publicaciones_llm_genera_predicciones_y_jsonl(tmp_path) -> None:
    """
    Verifica clasificación válida y persistencia de respuestas crudas.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {
            "text_id": ["t1", "t2"],
            "user_id": ["u1", "u2"],
            "title": ["risk", "calm"],
            "text": ["I want to die", "I am tired but safe"],
            "y": [1, 0],
        }
    )
    client = FakeGenerativeClient(
        [
            '{"label":"yes","score":0.91,"evidence":["want to die"],"justification":"risk"}',
            '{"label":"no","score":0.12,"evidence":["safe"],"justification":"no intent"}',
        ]
    )

    predicciones = clasificar_publicaciones_llm(
        frame,
        config,
        client,
        "qwen",
        tmp_path / "raw.jsonl",
    )

    assert predicciones["y_pred"].tolist() == [1, 0]
    assert predicciones["score"].tolist() == [0.91, 0.12]
    assert predicciones["error"].isna().all()
    assert len((tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_clasificar_publicaciones_llm_registra_error_tras_reintento(tmp_path) -> None:
    """
    Verifica que respuestas inválidas queden marcadas para revisión.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {"text_id": ["t1"], "user_id": ["u1"], "title": ["x"], "text": ["y"], "y": [1]}
    )
    client = FakeGenerativeClient(["not-json", '{"label":"maybe","score":0.2}'])

    predicciones = clasificar_publicaciones_llm(
        frame,
        config,
        client,
        "qwen",
        tmp_path / "raw.jsonl",
        max_attempts=2,
    )

    assert predicciones.loc[0, "y_pred"] is None
    assert isinstance(predicciones.loc[0, "error"], str)
    with pytest.raises(ValueError, match="respuestas LLM fallaron"):
        evaluar_y_guardar_llm(predicciones, config, tmp_path / "metrics.json")


def test_clasificar_publicaciones_llm_usa_prompt_few_shot(tmp_path) -> None:
    """
    Verifica que el modo few-shot agregue ejemplos al prompt.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {"text_id": ["t1"], "user_id": ["u1"], "title": ["x"], "text": ["y"], "y": [1]}
    )
    client = FakeGenerativeClient(
        ['{"label":"yes","score":0.8,"evidence":[],"justification":"risk"}']
    )

    predicciones = clasificar_publicaciones_llm(
        frame,
        config,
        client,
        "qwen",
        tmp_path / "raw.jsonl",
        prompt_mode="few_shot",
        few_shot_examples=[
            FewShotExample("yes-1", "yes", "I do not want to live anymore."),
            FewShotExample("no-1", "no", "I am sad but safe."),
        ],
    )

    assert predicciones.loc[0, "y_pred"] == 1
    assert "Few-shot examples" in client.prompts[0]
    assert "I am sad but safe." in client.prompts[0]


def test_seleccionar_ejemplos_few_shot_balancea_y_es_reproducible() -> None:
    """
    Verifica selección balanceada desde entrenamiento con semilla fija.
    """

    config = BaselineConfig()
    train = pd.DataFrame(
        {
            "text_id": [f"t{i}" for i in range(8)],
            "user_id": [f"u{i}" for i in range(8)],
            "title": ["title"] * 8,
            "text": [
                "suicidal ideation example with enough context to pass the length filter"
                for _ in range(4)
            ]
            + [
                "non suicidal support seeking example with enough context to pass filter"
                for _ in range(4)
            ],
            "y": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )

    first = seleccionar_ejemplos_few_shot(
        train,
        config,
        examples_per_class=2,
        random_state=7,
        min_chars=10,
    )
    second = seleccionar_ejemplos_few_shot(
        train,
        config,
        examples_per_class=2,
        random_state=7,
        min_chars=10,
    )

    assert [example.text_id for example in first] == [
        example.text_id for example in second
    ]
    assert [example.label for example in first].count("yes") == 2
    assert [example.label for example in first].count("no") == 2
    assert ejemplos_few_shot_a_json(first)[0]["text_id"].startswith("t")


def test_evaluar_y_guardar_llm_calcula_metricas(tmp_path) -> None:
    """
    Verifica evaluación de predicciones LLM válidas.
    """

    config = BaselineConfig()
    predicciones = pd.DataFrame(
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
            "score": [0.9, 0.1],
            "error": [None, None],
        }
    )

    metricas = evaluar_y_guardar_llm(predicciones, config, tmp_path / "metrics.json")
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert metricas["protocol_auc"] == pytest.approx(1.0)
    assert saved["roc_auc"] == pytest.approx(1.0)


def test_serializar_evidence_para_csv() -> None:
    """
    Verifica serialización estable de evidencia.
    """

    assert serializar_evidence_para_csv([["a", "b"], []]) == ['["a", "b"]', "[]"]
