"""Baseline del artículo: Random Forest + tf-idf (esquema 1, mejor modelo clásico: 0,802).

Réplica de src/classic_ML.ipynb de los autores (github.com/kevinmaiden7/Spanish_FakeNewsDetection):
- get_matrix(representation='tf-idf', vocabulary_length=40000, stemming=False, remove_stopwords=True)
  -> TfidfVectorizer(max_df=0.9, max_features=40000, min_df=0, use_idf=True), AJUSTADO CON TODO EL
  DATASET antes de separar train y test, como hicieron los autores.
- grid_search_cv: train_test_split(test_size=0.2) y GridSearchCV con un solo punto
  (n_estimators=500, max_features=50) validado con ShuffleSplit(n_splits=5, test_size=0.2).

Diferencia con los autores: ellos corrieron una sola partición sin semilla. Aquí se repite con
las semillas de params.yaml (random_state = semilla) para tener media y desviación.

Uso: python -m src.models.rf_articulo --variante sin_limpieza
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split

from src.data.replica import RAIZ, cargar_params, cargar_variante
from src.evaluation.metricas import guardar_json, metricas_split, resumir, textos_test_repetidos_en_train
from src.features.preprocesamiento_articulo import preprocesar_articulo


def main(variante: str) -> None:
    params = cargar_params()
    p = params["replica"]
    pre, rf = p["preprocesamiento"], p["rf"]

    df = cargar_variante(variante, params)
    textos = preprocesar_articulo(df["text"].tolist(), pre["quitar_stopwords"], pre["stemming"], pre["idioma"])

    vectorizador = TfidfVectorizer(max_df=rf["max_df"], max_features=rf["vocabulario"],
                                   min_df=rf["min_df"], use_idf=True)
    X = vectorizador.fit_transform(textos)
    y = df["label"].to_numpy()
    indices = np.arange(len(df))
    originales = df["text"].tolist()
    print(f"[rf_articulo] {variante}: {X.shape[0]} documentos, vocabulario {X.shape[1]}")

    por_semilla, predicciones = {}, []
    for semilla in p["semillas"]:
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y, indices, test_size=rf["test_size"], shuffle=True, random_state=semilla)

        cv = ShuffleSplit(n_splits=rf["cv_iteraciones"], test_size=rf["cv_test_size"], random_state=semilla)
        busqueda = GridSearchCV(RandomForestClassifier(random_state=semilla, n_jobs=-1),
                                {"n_estimators": [rf["n_estimators"]], "max_features": [rf["max_features"]]},
                                cv=cv, scoring="accuracy")
        busqueda.fit(X_tr, y_tr)
        y_pred = busqueda.predict(X_te)

        m = metricas_split(y_te, y_pred)
        m["accuracy_validacion"] = round(float(busqueda.best_score_), 4)
        m["std_validacion"] = round(float(busqueda.cv_results_["std_test_score"][0]), 4)
        m["n_train"], m["n_test"] = int(len(idx_tr)), int(len(idx_te))
        m["test_repetidos_en_train"] = textos_test_repetidos_en_train(originales, idx_tr, idx_te)
        por_semilla[f"semilla_{semilla}"] = m
        print(f"  semilla {semilla}: accuracy test {m['accuracy']} (validación {m['accuracy_validacion']})")

        prob = busqueda.predict_proba(X_te)[:, 1]
        predicciones.append(pd.DataFrame({"semilla": semilla, "id": df["id"].to_numpy()[idx_te],
                                          "label": y_te, "prob_falsa": prob.round(4), "pred": y_pred}))

    salida = RAIZ / p["carpeta_reportes"]
    resultado = {"modelo": "rf_tfidf", "variante": variante, "n_documentos": int(len(df)),
                 "vocabulario": int(X.shape[1]), **resumir(por_semilla), "por_semilla": por_semilla}
    guardar_json(resultado, salida / f"rf_{variante}.json")
    pd.concat(predicciones).to_csv(salida / f"predicciones_rf_{variante}.csv", index=False)
    print(f"[rf_articulo] accuracy media {resultado['accuracy_media']} ± {resultado['accuracy_std']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variante", required=True, choices=["sin_limpieza", "con_limpieza"])
    main(parser.parse_args().variante)