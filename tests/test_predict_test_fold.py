"""
Este módulo prueba el flujo de predicción para folds oficiales.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from reddit_mental_health.config import BaselineConfig


def _cargar_script_predict_test_fold() -> ModuleType:
    """
    Carga el script de predicción por ruta para probar sus helpers.
    """

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "predict_test_fold.py"
    spec = importlib.util.spec_from_file_location("predict_test_fold", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar scripts/predict_test_fold.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREDICT_TEST_FOLD = _cargar_script_predict_test_fold()


def _crear_train_csv(path: Path) -> None:
    """
    Crea datos mínimos de construcción para entrenar en pruebas.
    """

    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "text_id": ["tr1", "tr2", "tr3", "tr4"],
            "title": ["riesgo ayuda", "calma vida", "riesgo alerta", "calma estable"],
            "text": ["riesgo ayuda", "calma vida", "riesgo alerta", "calma estable"],
            "is_suicide": ["yes", "no", "yes", "no"],
        }
    ).to_csv(path, index=False)


def _crear_test_csv(path: Path, include_label: bool = True) -> None:
    """
    Crea un fold de prueba mínimo con etiquetas opcionales.
    """

    datos = {
        "user_id": ["u5", "u6"],
        "text_id": ["te1", "te2"],
        "title": ["riesgo ayuda", "calma vida"],
        "text": ["riesgo alerta", "calma estable"],
    }
    if include_label:
        datos["is_suicide"] = ["yes", "no"]
    pd.DataFrame(datos).to_csv(path, index=False)


def _args(tmp_path: Path, evaluate_if_labeled: bool, include_label: bool) -> argparse.Namespace:
    """
    Construye argumentos CLI mínimos para ejecutar el flujo en pruebas.
    """

    train_input = tmp_path / "data_train.csv"
    test_input = tmp_path / "data_test_fold1.csv"
    _crear_train_csv(train_input)
    _crear_test_csv(test_input, include_label=include_label)
    return argparse.Namespace(
        train_input=train_input,
        test_input=test_input,
        classifier_name="logistic_regression",
        feature_config_name="word_unigram",
        predictions_out=tmp_path / "predictions.csv",
        metrics_out=tmp_path / "metrics.json",
        metadata_out=tmp_path / "metadata.json",
        model_out=tmp_path / "model.joblib",
        figures_dir=tmp_path / "figures",
        dashboard_out=tmp_path / "dashboard.html",
        no_model=True,
        evaluate_if_labeled=evaluate_if_labeled,
        no_dashboard=True,
    )


def test_construir_predicciones_fold_mapea_label_pred() -> None:
    """
    Verifica que label_pred traduzca 0/1 a no/yes.
    """

    config = BaselineConfig()
    test_data = pd.DataFrame({"text_id": ["t1", "t2"], "user_id": ["u1", "u2"]})

    predicciones = PREDICT_TEST_FOLD.construir_predicciones_fold(
        test_data,
        y_pred=[0, 1],
        score=[0.2, 0.8],
        config=config,
        include_y_true=False,
    )

    assert predicciones["label_pred"].tolist() == ["no", "yes"]
    assert predicciones.columns.tolist() == [
        "text_id",
        "user_id",
        "y_pred",
        "label_pred",
        "score",
    ]


def test_prediccion_sin_evaluacion_ignora_etiquetas_presentes(tmp_path) -> None:
    """
    Verifica que las etiquetas presentes se ignoren por default.
    """

    args = _args(tmp_path, evaluate_if_labeled=False, include_label=True)

    PREDICT_TEST_FOLD.ejecutar_prediccion(args)
    predicciones = pd.read_csv(args.predictions_out)
    metadata = json.loads(args.metadata_out.read_text(encoding="utf-8"))

    assert "y_true" not in predicciones.columns
    assert not args.metrics_out.exists()
    assert metadata["labels_present_in_test_fold"] is True
    assert metadata["labels_used_for_evaluation"] is False


def test_prediccion_sin_etiquetas_funciona(tmp_path) -> None:
    """
    Verifica que un fold sin is_suicide genere predicciones.
    """

    args = _args(tmp_path, evaluate_if_labeled=False, include_label=False)

    PREDICT_TEST_FOLD.ejecutar_prediccion(args)
    predicciones = pd.read_csv(args.predictions_out)
    metadata = json.loads(args.metadata_out.read_text(encoding="utf-8"))

    assert predicciones.columns.tolist() == [
        "text_id",
        "user_id",
        "y_pred",
        "label_pred",
        "score",
    ]
    assert metadata["labels_present_in_test_fold"] is False


def test_metricas_solo_se_generan_con_evaluacion_explicita(tmp_path) -> None:
    """
    Verifica que las métricas se guarden solo con evaluación explícita.
    """

    args = _args(tmp_path, evaluate_if_labeled=True, include_label=True)

    PREDICT_TEST_FOLD.ejecutar_prediccion(args)
    predicciones = pd.read_csv(args.predictions_out)
    metricas = json.loads(args.metrics_out.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata_out.read_text(encoding="utf-8"))

    assert "y_true" in predicciones.columns
    assert {
        "roc_auc",
        "protocol_auc",
        "true_positive_rate",
        "false_positive_rate",
        "recall",
        "precision",
        "f1",
        "confusion_matrix",
    }.issubset(metricas)
    assert metadata["labels_present_in_test_fold"] is True
    assert metadata["labels_used_for_evaluation"] is True
    assert metadata["metrics_out"] == str(args.metrics_out)


def test_evaluacion_falla_si_no_hay_etiquetas(tmp_path) -> None:
    """
    Verifica que la evaluación explícita falle sin etiquetas reales.
    """

    args = _args(tmp_path, evaluate_if_labeled=True, include_label=False)

    with pytest.raises(SystemExit, match="no contiene etiquetas"):
        PREDICT_TEST_FOLD.ejecutar_prediccion(args)
