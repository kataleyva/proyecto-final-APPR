"""Pruebas de la réplica del artículo. Usan datos sintéticos; no descargan BETO."""
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.features.embeddings_beto import construir_entradas, construir_tokens
from src.features.preprocesamiento_articulo import normalizar_texto, preprocesar_articulo
from src.models.beto_lstm_articulo import LSTMArticulo


class TokenizadorFalso:
    vocab = {"[CLS]": 4, "[SEP]": 5}

    def tokenize(self, texto):
        return texto.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, 10 + len(t)) for t in tokens]


def test_normalizacion_igual_a_los_autores():
    # minúsculas y sin puntuación ASCII; ¿ ¡ y tildes se quedan, como en text_normalization
    assert normalizar_texto("¡Hola, MUNDO! ¿Qué tal?") == "¡hola mundo ¿qué tal"
    assert normalizar_texto("*NUMBER* años") == "number años"


def test_preprocesamiento_quita_stopwords():
    assert preprocesar_articulo(["El perro de la casa"]) == ["perro casa"]


def test_tokens_cortados_como_los_autores():
    tok = TokenizadorFalso()
    tokens = construir_tokens(" ".join(["a"] * 500), tok, 150)
    assert len(tokens) == 150 and tokens[0] == "[CLS]" and tokens[-1] == "[SEP]"
    ids, mascaras, segmentos = construir_entradas([tokens, ["[CLS]", "b", "[SEP]"]], tok, 150)
    assert ids.shape == (2, 150)
    assert mascaras[1].sum() == 3 and ids[1, 3:].sum() == 0  # relleno con 0
    assert segmentos.sum() == 0


def test_particion_igual_con_o_sin_columnas_extra():
    # train_test_split solo depende de n y random_state: mismas particiones que los autores
    n = 2571
    a_tr, a_te = train_test_split(np.arange(n), test_size=0.1, random_state=3)
    _, _, b_tr, b_te = train_test_split(np.zeros((n, 2)), np.arange(n), test_size=0.1, random_state=3)
    assert (a_te == b_te).all() and len(a_te) == 258


def test_lstm_inicializada_como_keras():
    modelo = LSTMArticulo(768, 8, 0.5)
    assert torch.all(modelo.lstm.bias_ih_l0[8:16] == 1)
    assert torch.all(modelo.lstm.bias_hh_l0 == 0)
    salida = modelo(torch.randn(3, 150, 768))
    assert salida.shape == (3,) and torch.all((salida > 0) & (salida < 1))