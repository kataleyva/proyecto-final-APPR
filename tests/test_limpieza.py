# Pruebas del pipeline de limpieza
import pandas as pd
import pytest

from src.data.limpieza import (
    limpiar, normalizar_espacios, normalizar_unicode, quitar_emojis_y_simbolos,
    quitar_huellas_fuente, quitar_otros_alfabetos, quitar_urls, reparar_codificacion,
    reparar_homoglifos, separar_frases_pegadas, unificar_numeros,
)


# Caracteres extraños reparados
@pytest.mark.parametrize("entrada, esperado", [
    ("don\x92t", "don’t"),                              # C1 (cp1252 leído como Latin-1)
    ("\x93scrambled\x94", "“scrambled”"),
    ("isnâ€™t", "isn’t"),                              # UTF-8 leído como cp1252
    ("cafÃ©", "café"),
    ("AT&amp;T", "AT&T"),                               # entidad HTML
    ("hack &mdsh; leaving", "hack   leaving"),          # entidad mal escrita eliminada
])
def test_reparar_codificacion(entrada, esperado):
    assert reparar_codificacion(entrada) == esperado


def test_normalizar_unicode_quita_invisibles_y_espacios_raros():
    assert normalizar_unicode("a\U000000A0b\U0000200Bc\U000000ADd") == "a bcd"   # NBSP, ancho cero, soft hyphen


def test_homoglifos_en_espanol():
    assert reparar_homoglifos("Pese a c\U0000043Entar c\U0000043En", "es") == "Pese a contar con"


def test_homoglifos_en_ingles_no_toca_texto_ruso_aislado():
    assert reparar_homoglifos("c\U0000043Entact Москва", "en") == "contact Москва"


# Mantener ortografía
def test_respeta_tildes_enes_y_signos():
    texto = "¿Quién dijo que el año pasado el niño ganó? ¡Jamás!"
    limpio = texto
    for paso in (reparar_codificacion, normalizar_unicode, quitar_emojis_y_simbolos,
                 quitar_otros_alfabetos, separar_frases_pegadas, normalizar_espacios):
        limpio = paso(limpio)
    assert limpio == texto


def test_no_pasa_a_minusculas():
    assert normalizar_espacios("BREAKING News") == "BREAKING News"


# Emojis
def test_quita_emojis_banderas_y_dingbats():
    assert normalizar_espacios(quitar_emojis_y_simbolos("Hola 📌 🇺🇸 ✓ ❖ adiós 🙂")) == "Hola adiós"


# Contexto: huellas de fuente y formato
def test_quita_firma_reuters():
    assert quitar_huellas_fuente("WASHINGTON (Reuters) - The U.S. said").strip() == "The U.S. said"


def test_quita_credito_de_foto_final():
    assert quitar_huellas_fuente("He left.Featured image via Getty Images").strip() == "He left."


def test_quita_urls_incluso_pegadas():
    assert "pic.twitter" not in quitar_urls("2017pic.twitter.com/abc y https://t.co/x")


@pytest.mark.parametrize("entrada, esperado", [
    ("it off.Watch:", "it off. Watch:"),
    ("FilesSUPPORT", "Files SUPPORT"),
    ("2017The", "2017 The"),
    ("iPhone y McDonald", "iPhone y McDonald"),         # no rompe marcas
    ("EE.UU. dijo", "EE.UU. dijo"),                     # no rompe siglas
])
def test_separar_frases_pegadas(entrada, esperado):
    assert separar_frases_pegadas(entrada) == esperado


def test_unificar_numeros():
    assert unificar_numeros("cuesta 25 pesos, 1.500 en total, CO2") == "cuesta *NUMBER* pesos, *NUMBER* en total, CO2"


def test_otros_alfabetos():
    assert quitar_otros_alfabetos("dijo 我 shalom שלום 1.º") == "dijo  shalom  1.º"


# Pipeline completo 
PARAMS = {"min_caracteres": 5, "quitar_tretiakov_de_posadas_v2": True, "pasos": {}}


def _df(filas):
    return pd.DataFrame(filas, columns=["id", "fuente", "archivo", "idioma", "uso", "texto", "label"])


def test_pipeline_quita_duplicados_creados_por_la_limpieza_y_valida():
    df = _df([
        ("a", "bisaillon", "f", "en", "modelo_b", "Trump said no.Featured image via Flickr", 1),
        ("b", "bisaillon", "f", "en", "modelo_b", "Trump said no.Featured image via Getty", 1),
        ("c", "bisaillon", "f", "en", "modelo_b", "   ", 0),
    ])
    limpio, reporte = limpiar(df, PARAMS)
    assert len(limpio) == 1                       # el vacío y el duplicado "nuevo" se van
    assert reporte["despues_de_limpiar"]["duplicados"] == 1
    assert all(v is True or v == 0 for v in reporte["validacion"].values())


def test_pipeline_quita_tretiakov_dentro_de_posadas_v2():
    texto = "El gobierno anunció hoy una medida muy polémica para todos los ciudadanos"
    df = _df([
        ("t1", "tretiakov", "f", "es", "replica", texto, 1),
        ("v1", "posadas_v2", "f", "es", "generalizacion", texto + " y mucho más contenido nuevo.", 1),
    ])
    limpio, reporte = limpiar(df, PARAMS)
    assert list(limpio["id"]) == ["t1"]
    assert reporte["tretiakov_quitadas_de_posadas_v2"] == 1


def test_un_paso_se_puede_apagar():
    df = _df([("a", "ukani", "f", "en", "modelo_b", "Hola 📌 mundo", 0)])
    limpio, reporte = limpiar(df, {**PARAMS, "pasos": {"quitar_emojis_y_simbolos": False}})
    assert "📌" in limpio.loc[0, "texto_limpio"]
    assert reporte["pasos"]["quitar_emojis_y_simbolos"] == "apagado"


def test_quita_reuters_en_medio_del_texto():
    assert "(Reuters)" not in quitar_huellas_fuente("He said: WASHINGTON (Reuters) CIA Director")


def test_credito_de_foto_pegado_a_un_numero():
    assert quitar_huellas_fuente("December 2017Featured image via Getty").strip() == "December 2017"


def test_no_rompe_usuarios_de_twitter():
    assert separar_frases_pegadas("(@TimJHanrahan) y FilesSUPPORT") == "(@TimJHanrahan) y Files SUPPORT"