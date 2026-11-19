"""Métricas comunes para los modelos de la réplica (1 = noticia falsa es la clase positiva)."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def metricas_split(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        # Como en la matriz de confusión del artículo: % de falsas y de reales bien clasificadas
        "acierto_falsas": round(float(tp / (tp + fn)) if tp + fn else 0.0, 4),
        "acierto_reales": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def textos_test_repetidos_en_train(textos, idx_train, idx_test) -> int:
    """Cuántas noticias del test tienen un texto idéntico en train (fuga por duplicados)."""
    vistos = {textos[i] for i in idx_train}
    return int(sum(textos[i] in vistos for i in idx_test))


def resumir(por_semilla: dict) -> dict:
    claves = ["accuracy", "precision", "recall", "f1", "acierto_falsas", "acierto_reales"]
    resumen = {}
    for clave in claves:
        valores = np.array([m[clave] for m in por_semilla.values()])
        resumen[f"{clave}_media"] = round(float(valores.mean()), 4)
        resumen[f"{clave}_std"] = round(float(valores.std(ddof=1)) if len(valores) > 1 else 0.0, 4)
    return resumen


def guardar_json(datos: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)