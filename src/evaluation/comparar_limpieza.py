"""Compara el mismo modelo del artículo con y sin nuestra limpieza.

Lee reports/replica/{rf,beto_lstm}_{sin_limpieza,con_limpieza}.json y escribe
reports/replica/comparacion_limpieza.json y comparacion_limpieza.md.

Con 5 particiones por variante, la prueba t de Welch es solo orientativa: con tan pocas
repeticiones, diferencias menores a ~2 desviaciones estándar no se pueden distinguir del azar.

Uso: python -m src.evaluation.comparar_limpieza
"""
import json

import numpy as np
from scipy import stats

from src.data.replica import RAIZ, cargar_params
from src.evaluation.metricas import guardar_json

# Valores de los autores (artículo y src/classic_ML.ipynb, src/BETO.ipynb)
REFERENCIA = {
    "rf_tfidf": "0,802 (una sola partición 80/20, sin semilla)",
    "beto_lstm": "0,769 de media en 5 particiones; 0,80 en la mejor (la que reporta el artículo)",
}
NOMBRES = {"rf_tfidf": "Random Forest + tf-idf", "beto_lstm": "BETO congelado + LSTM"}


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = cargar_params()["replica"]
    carpeta = RAIZ / p["carpeta_reportes"]
    filas, resultado = [], {}
    for prefijo, modelo in [("rf", "rf_tfidf"), ("beto_lstm", "beto_lstm")]:
        sin = cargar(carpeta / f"{prefijo}_sin_limpieza.json")
        con = cargar(carpeta / f"{prefijo}_con_limpieza.json")
        n_sin, n_con = sin["n_documentos"], con["n_documentos"]
        acc_sin = np.array([m["accuracy"] for m in sin["por_semilla"].values()])
        acc_con = np.array([m["accuracy"] for m in con["por_semilla"].values()])
        t = stats.ttest_ind(acc_con, acc_sin, equal_var=False)
        fuga = sum(m["test_repetidos_en_train"] for m in sin["por_semilla"].values())
        resultado[modelo] = {
            "accuracy_sin_limpieza": sin["accuracy_media"], "std_sin_limpieza": sin["accuracy_std"],
            "accuracy_con_limpieza": con["accuracy_media"], "std_con_limpieza": con["accuracy_std"],
            "diferencia": round(float(acc_con.mean() - acc_sin.mean()), 4),
            "p_valor_welch": round(float(t.pvalue), 4),
            "f1_sin_limpieza": sin["f1_media"], "f1_con_limpieza": con["f1_media"],
            "test_repetidos_en_train_sin_limpieza": int(fuga),
        }
        r = resultado[modelo]
        filas.append(f"| {NOMBRES[modelo]} | {r['accuracy_sin_limpieza']:.3f} ± {r['std_sin_limpieza']:.3f} | "
                     f"{r['accuracy_con_limpieza']:.3f} ± {r['std_con_limpieza']:.3f} | {r['diferencia']:+.3f} | "
                     f"{r['p_valor_welch']:.3f} | {REFERENCIA[modelo]} |")

    guardar_json(resultado, carpeta / "comparacion_limpieza.json")
    md = ["# Réplica del artículo: con y sin nuestra limpieza", "",
          f"Mismo modelo, mismo preprocesamiento y mismas semillas ({p['semillas']}). Solo cambian los datos:",
          f"{n_sin} textos crudos (idénticos a los de los autores) contra {n_con} textos limpios.", "",
          "| Modelo | Accuracy sin limpieza | Accuracy con limpieza | Diferencia | p (Welch) | Autores |",
          "|---|---|---|---|---|---|", *filas, "",
          "Accuracy = media ± desviación estándar en las particiones. p < 0,05 sugiere una diferencia real; "
          "con 5 particiones es solo orientativo.", ""]
    for modelo, r in resultado.items():
        md.append(f"- {NOMBRES[modelo]}: en la versión sin limpieza, {r['test_repetidos_en_train_sin_limpieza']} "
                  "noticias de test (sumando las particiones) tenían un texto idéntico en train.")
    (carpeta / "comparacion_limpieza.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()