"""Modelo A del artículo: BETO congelado + LSTM (esquema 2, mejor modelo profundo en español).

Réplica de create_model_RNN y del bucle final de src/BETO.ipynb de los autores
(github.com/kevinmaiden7/Spanish_FakeNewsDetection), que dio test_acc_avg = 0,769 en 5 particiones
y 0,8023 en la mejor (la que reporta el artículo):
- LSTM(units=8, return_sequences=False), sin regularizadores (l2 = 0)
- Dropout(0.5) -> Dense(1, sigmoid)
- loss binary_crossentropy, optimizer 'adam', batch_size 32, 25 épocas, shuffle, sin early stopping
- train_test_split(test_size=0.1, random_state=i) para i = 0..4

Los autores usaban Keras (TensorFlow 2.3). Aquí se usa PyTorch con los valores por defecto de Keras:
inicialización glorot_uniform (entrada y Dense), orthogonal (recurrente), sesgo 0 con sesgo de la
puerta de olvido = 1 (unit_forget_bias), Adam lr=0.001, betas (0.9, 0.999), epsilon=1e-7, y la
pérdida recorta las probabilidades a [1e-7, 1 - 1e-7] como Keras.

Uso: python -m src.models.beto_lstm_articulo --variante sin_limpieza
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from src.data.replica import RAIZ, cargar_params
from src.evaluation.metricas import guardar_json, metricas_split, resumir, textos_test_repetidos_en_train


class LSTMArticulo(nn.Module):
    def __init__(self, entrada: int = 768, unidades: int = 8, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(entrada, unidades, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.salida = nn.Linear(unidades, 1)
        self._inicializar_como_keras(unidades)

    def _inicializar_como_keras(self, unidades: int) -> None:
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)   # kernel: glorot_uniform
        nn.init.orthogonal_(self.lstm.weight_hh_l0)       # recurrent_kernel: orthogonal
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        with torch.no_grad():                              # unit_forget_bias=True (orden i, f, g, o)
            self.lstm.bias_ih_l0[unidades: 2 * unidades] = 1.0
        nn.init.xavier_uniform_(self.salida.weight)
        nn.init.zeros_(self.salida.bias)

    def forward(self, x):
        _, (h, _) = self.lstm(x)                           # many-to-one: último estado oculto
        return torch.sigmoid(self.salida(self.dropout(h[-1]))).squeeze(-1)


def perdida_keras(prob, y, eps: float = 1e-7):
    prob = prob.clamp(eps, 1 - eps)
    return nn.functional.binary_cross_entropy(prob, y)


def predecir(modelo, X, indices, lote: int = 64) -> np.ndarray:
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(indices), lote):
            bloque = torch.from_numpy(X[indices[inicio: inicio + lote]].astype(np.float32))
            salidas.append(modelo(bloque).numpy())
    return np.concatenate(salidas)


def entrenar(X, y, idx_train, lstm: dict, semilla: int):
    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)
    modelo = LSTMArticulo(X.shape[2], lstm["unidades"], lstm["dropout"])
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lstm["lr"], betas=(0.9, 0.999), eps=1e-7)
    historial = []
    for epoca in range(lstm["epocas"]):
        modelo.train()
        orden = idx_train[torch.randperm(len(idx_train), generator=generador).numpy()]
        perdida_total, aciertos = 0.0, 0
        for inicio in range(0, len(orden), lstm["batch_size"]):
            lote = orden[inicio: inicio + lstm["batch_size"]]
            xb = torch.from_numpy(X[lote].astype(np.float32))
            yb = torch.from_numpy(y[lote].astype(np.float32))
            optimizador.zero_grad()
            prob = modelo(xb)
            perdida = perdida_keras(prob, yb)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(lote)
            aciertos += int(((prob > 0.5).float() == yb).sum())
        historial.append({"epoca": epoca + 1, "loss": round(perdida_total / len(orden), 4),
                          "accuracy": round(aciertos / len(orden), 4)})
    return modelo, historial


def main(variante: str) -> None:
    params = cargar_params()
    p = params["replica"]
    lstm = p["lstm"]

    datos = np.load(RAIZ / p["beto"]["carpeta_embeddings"] / f"{variante}.npz")
    X, y, ids, textos = datos["X"], datos["y"], datos["ids"], datos["textos"].tolist()
    indices = np.arange(len(y))
    print(f"[beto_lstm] {variante}: X {X.shape}")

    por_semilla, predicciones = {}, []
    for semilla in p["semillas"]:
        idx_tr, idx_te = train_test_split(indices, test_size=lstm["test_size"], random_state=semilla)
        modelo, historial = entrenar(X, y, idx_tr, lstm, semilla)
        prob = predecir(modelo, X, idx_te)
        y_pred = (prob > 0.5).astype(int)

        m = metricas_split(y[idx_te], y_pred)
        m["loss_train_final"] = historial[-1]["loss"]
        m["accuracy_train_final"] = historial[-1]["accuracy"]
        m["n_train"], m["n_test"] = int(len(idx_tr)), int(len(idx_te))
        m["test_repetidos_en_train"] = textos_test_repetidos_en_train(textos, idx_tr, idx_te)
        por_semilla[f"semilla_{semilla}"] = m
        print(f"  semilla {semilla}: accuracy test {m['accuracy']} (train {m['accuracy_train_final']})")
        predicciones.append(pd.DataFrame({"semilla": semilla, "id": ids[idx_te], "label": y[idx_te],
                                          "prob_falsa": prob.round(4), "pred": y_pred}))

    salida = RAIZ / p["carpeta_reportes"]
    resultado = {"modelo": "beto_lstm", "variante": variante, "n_documentos": int(len(y)),
                 **resumir(por_semilla), "por_semilla": por_semilla}
    guardar_json(resultado, salida / f"beto_lstm_{variante}.json")
    pd.concat(predicciones).to_csv(salida / f"predicciones_beto_lstm_{variante}.csv", index=False)
    print(f"[beto_lstm] accuracy media {resultado['accuracy_media']} ± {resultado['accuracy_std']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variante", required=True, choices=["sin_limpieza", "con_limpieza"])
    main(parser.parse_args().variante)