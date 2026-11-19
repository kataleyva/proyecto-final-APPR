"""Preprocesamiento del artículo, copiado del código de los autores.

Fuente: github.com/kevinmaiden7/Spanish_FakeNewsDetection, src/data_preprocessing.py
(funciones text_normalization, remove_stop_words y apply_stemming) y src/BETO.ipynb
(función normalize_and_tokenize_data). Solo cambia la forma (funciones sobre listas en vez
de modificar un DataFrame fila por fila); el resultado es el mismo texto.
"""
import re
import string

import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

_PUNTUACION = re.compile("[%s]" % re.escape(string.punctuation))


def descargar_recursos_nltk() -> None:
    """Los autores hacían nltk.download('punkt') y ('stopwords'). Las versiones nuevas de
    NLTK usan 'punkt_tab' para word_tokenize."""
    for recurso in ("punkt_tab", "stopwords"):
        nltk.download(recurso, quiet=True)


def normalizar_texto(texto: str) -> str:
    """text_normalization: minúsculas y quitar la puntuación ASCII (string.punctuation).
    Ojo: los signos ¿ ¡ « » y las tildes NO se quitan, igual que en el original."""
    return _PUNTUACION.sub("", texto.lower())


def quitar_stopwords(texto: str, stopwords: set) -> str:
    """remove_stop_words(get_tokenize=False): word_tokenize (idioma por defecto, como los
    autores), filtrar stop words de NLTK y volver a unir con espacios."""
    return " ".join(palabra for palabra in word_tokenize(texto) if palabra not in stopwords)


def aplicar_stemming(texto: str, stemmer: SnowballStemmer) -> str:
    """apply_stemming: Snowball, palabra por palabra."""
    return " ".join(stemmer.stem(palabra) for palabra in texto.split())


def preprocesar_articulo(textos, quitar_stop_words: bool = True, stemming: bool = False,
                         idioma: str = "spanish") -> list:
    """Mismo orden que get_matrix de los autores: normalizar, stop words, stemming."""
    descargar_recursos_nltk()
    stopwords = set(nltk.corpus.stopwords.words(idioma))
    stemmer = SnowballStemmer(idioma) if stemming else None
    resultado = []
    for texto in textos:
        texto = normalizar_texto(texto)
        if quitar_stop_words:
            texto = quitar_stopwords(texto, stopwords)
        if stemmer is not None:
            texto = aplicar_stemming(texto, stemmer)
        resultado.append(texto)
    return resultado