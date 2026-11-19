"""Embeddings de BETO congelado, como en src/BETO.ipynb de los autores.

Réplica de normalize_and_tokenize_data, get_ids, get_masks y get_segments:
1. text_normalization + remove_stop_words (español).
2. Tokenizador de BETO uncased construido desde vocab.txt con do_lower_case=True
   (los autores hacían AutoTokenizer.from_pretrained('.../vocab.txt', do_lower_case=True);
   eso también quita las tildes).
3. [CLS] + las primeras 148 piezas + [SEP]  (máximo 150 tokens).
4. Relleno con el id 0 hasta 150, máscara de atención 1/0 y segmentos en 0.
5. BETO congelado -> last_hidden_state (150 x 768 por noticia). La LSTM lee las 150 posiciones,
   incluidas las de relleno, igual que en Keras (la salida de BETO no lleva máscara).

Diferencia con los autores: ellos pasaban el texto por BETO en cada época dentro de Keras
(~12 minutos por época). Como BETO está congelado, aquí se calcula una sola vez y se guarda.
La única consecuencia es que no hay dropout interno de BETO durante el entrenamiento.

Uso: python -m src.features.embeddings_beto --variante sin_limpieza
"""
import argparse
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, BertTokenizer

from src.data.replica import RAIZ, cargar_params, cargar_variante
from src.features.preprocesamiento_articulo import preprocesar_articulo


def cargar_tokenizador(modelo: str) -> BertTokenizer:
    if os.path.isdir(modelo):
        vocab = os.path.join(modelo, "vocab.txt")
    else:
        vocab = hf_hub_download(modelo, "vocab.txt")
    # En transformers 5 el argumento se llama "vocab" (con "vocab_file" se ignora en silencio
    # y todo sale [UNK]). do_lower_case=True sin strip_accents también quita tildes, como en 2020.
    tokenizador = BertTokenizer(vocab=vocab, do_lower_case=True)
    if len(tokenizador.get_vocab()) < 1000:
        raise ValueError(f"El vocabulario de BETO no cargó bien: {len(tokenizador.get_vocab())} piezas")
    return tokenizador


def construir_tokens(texto: str, tokenizador, max_tokens: int) -> list:
    """Igual que el bucle de normalize_and_tokenize_data: corta en max_tokens - 2 piezas."""
    piezas = tokenizador.tokenize(texto)[: max_tokens - 2]
    return ["[CLS]"] + piezas + ["[SEP]"]


def construir_entradas(tokens_por_texto: list, tokenizador, max_tokens: int):
    n = len(tokens_por_texto)
    ids = np.zeros((n, max_tokens), dtype=np.int64)       # relleno con 0, como get_ids
    mascaras = np.zeros((n, max_tokens), dtype=np.int64)
    for i, tokens in enumerate(tokens_por_texto):
        ids[i, : len(tokens)] = tokenizador.convert_tokens_to_ids(tokens)
        mascaras[i, : len(tokens)] = 1
    segmentos = np.zeros((n, max_tokens), dtype=np.int64)  # get_segments da todo 0
    return ids, mascaras, segmentos


def main(variante: str) -> None:
    params = cargar_params()
    p = params["replica"]
    pre, beto = p["preprocesamiento"], p["beto"]

    df = cargar_variante(variante, params)
    textos = preprocesar_articulo(df["text"].tolist(), quitar_stop_words=pre["quitar_stopwords"],
                                  stemming=False, idioma=pre["idioma"])

    tokenizador = cargar_tokenizador(beto["modelo"])
    tokens = [construir_tokens(t, tokenizador, beto["max_tokens"]) for t in textos]
    ids, mascaras, segmentos = construir_entradas(tokens, tokenizador, beto["max_tokens"])
    cortados = sum(len(tokenizador.tokenize(t)) > beto["max_tokens"] - 2 for t in textos)
    piezas = [pieza for t in tokens for pieza in t[1:-1]]
    pct_unk = 100 * sum(pieza == tokenizador.unk_token for pieza in piezas) / max(len(piezas), 1)
    print(f"[embeddings_beto] {variante}: {pct_unk:.2f} % de piezas [UNK]")
    if pct_unk > 5:
        raise ValueError(f"Demasiadas piezas [UNK] ({pct_unk:.1f} %): revisa el tokenizador")

    torch.manual_seed(0)
    modelo = AutoModel.from_pretrained(beto["modelo"])
    modelo.eval()  # congelado: no se entrena ni se usa dropout
    tipo = np.float16 if beto["guardar_float16"] else np.float32
    X = np.zeros((len(df), beto["max_tokens"], modelo.config.hidden_size), dtype=tipo)

    lote = beto["lote"]
    with torch.no_grad():
        for inicio in range(0, len(df), lote):
            fin = min(inicio + lote, len(df))
            salida = modelo(input_ids=torch.from_numpy(ids[inicio:fin]),
                            attention_mask=torch.from_numpy(mascaras[inicio:fin]),
                            token_type_ids=torch.from_numpy(segmentos[inicio:fin]))
            X[inicio:fin] = salida.last_hidden_state.numpy().astype(tipo)
            if (inicio // lote) % 20 == 0:
                print(f"[embeddings_beto] {variante}: {fin}/{len(df)}")

    destino = RAIZ / beto["carpeta_embeddings"]
    destino.mkdir(parents=True, exist_ok=True)
    np.savez(destino / f"{variante}.npz", X=X, y=df["label"].to_numpy(),
             ids=df["id"].to_numpy().astype(str), textos=df["text"].to_numpy().astype(str))
    info = {"variante": variante, "n_documentos": int(len(df)), "forma": list(X.shape),
            "textos_cortados_a_max_tokens": int(cortados), "porcentaje_unk": round(pct_unk, 3),
            "tokens_reales_promedio": round(float(mascaras.sum(axis=1).mean()), 1)}
    with open(destino / f"{variante}_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[embeddings_beto] listo: {info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variante", required=True, choices=["sin_limpieza", "con_limpieza"])
    main(parser.parse_args().variante)
