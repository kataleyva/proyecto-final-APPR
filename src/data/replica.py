"""Carga los datos de la réplica de Martínez-Gallego et al. (2021) en sus dos variantes.

- sin_limpieza: los 2571 textos tal como los usaron los autores
  (data/interim/noticias_unidas_local.csv, columna "texto", uso == "replica").
  Verificado el 2026-11-18: son idénticos y están en el mismo orden que
  data/Merged/spanish_dataset.csv del repositorio de los autores.
- con_limpieza: los mismos documentos después de nuestro pipeline de limpieza
  (data/processed/noticias_limpias.csv, columna "texto_limpio", uso == "replica").
  Quedan 2531, porque la limpieza quitó duplicados y textos vacíos o cortos.

El orden de las filas se conserva tal cual viene en el archivo, porque las particiones
train/test dependen del orden (igual que en el código de los autores).
"""
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]


def cargar_params(ruta: Path = RAIZ / "params.yaml") -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_variante(nombre: str, params: dict) -> pd.DataFrame:
    """Devuelve un DataFrame con las columnas id, text y label (1 = falsa, 0 = real)."""
    config = params["replica"]["variantes"][nombre]
    df = pd.read_csv(RAIZ / config["archivo"])
    df = df[df["uso"] == params["replica"]["uso"]]
    df = df[["id", config["columna_texto"], "label"]].rename(columns={config["columna_texto"]: "text"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df