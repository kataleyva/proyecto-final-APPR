
"""Pipeline de limpieza de noticias

- Reparar antes de borrar (mojibake, homoglifos, espacios raros).
- Respetar la ortografía: no se corrige nada, no se quitan tildes ni eñes.
- No se pasa a minúsculas ni se quitan stop words: eso es del baseline (fase 4).

"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

# Mojibake UTF-8 leído como Windows-1252 ("â€™" en vez de "’", "Ã©" en vez de "é") y  Windows-1252 leído como Latin-1 (controles C1 invisibles, "\x92" en vez de "’")
_CP1252_ALTOS = "€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ"
MOJIBAKE_UTF8_RE = re.compile(f"[ÂÃâ][\x80-\xbf{_CP1252_ALTOS}]{{1,2}}")
C1_RE = re.compile("[\x80-\x9f]")
HTML_ENTIDAD_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});")
# Restos de doble mojibake que no se pueden revertir con una sola decodificación
MOJIBAKE_RESTOS = {"â\x80Ś": "…", "â\x82Ź": "€", "â€Ś": "…", "â‚Ź": "€"}

URL_RE = re.compile(
    r"https?://\S+|www\.\S+"
    r"|(?<![A-Za-z.])(?:pic\.twitter\.com|t\.co|bit\.ly|youtu\.be|goo\.gl|tmsnrt\.rs|t\.me)/\S*"
)
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF]"      # pictogramas, banderas, tonos de piel
    "|[\U00002300-\U000023FF\U00002600-\U000027BF\U00002B00-\U00002BFF]"  # dingbats y símbolos (✓ ♦ ❖ ☢)
    "|[\U0000FE0E\U0000FE0F]"  # selectores de variación (invisibles)
)

# Letras cirílicas y griegas que se ven idénticas a letras latinas
HOMOGLIFOS = str.maketrans({  # código Unicode de la letra cirílica (04xx) o griega (03xx) -> letra latina
    "\U00000430": "a", "\U00000435": "e", "\U0000043E": "o", "\U00000440": "p", "\U00000441": "c", "\U00000443": "y",
    "\U00000445": "x", "\U00000456": "i", "\U00000458": "j", "\U00000455": "s", "\U00000501": "d", "\U000004BB": "h",
    "\U000004CF": "l", "\U00000410": "A", "\U00000412": "B", "\U00000415": "E", "\U0000041A": "K", "\U0000041C": "M",
    "\U0000041D": "H", "\U0000041E": "O", "\U00000420": "P", "\U00000421": "C", "\U00000422": "T", "\U00000425": "X",
    "\U00000423": "Y", "\U00000406": "I", "\U00000408": "J", "\U00000405": "S", "\U000003BF": "o", "\U0000039F": "O",
    "\U000003B1": "a", "\U000003BD": "v", "\U000003C1": "p", "\U000003C4": "t", "\U00000391": "A", "\U00000392": "B",
    "\U00000395": "E", "\U00000397": "H", "\U00000399": "I", "\U0000039A": "K", "\U0000039C": "M", "\U0000039D": "N",
    "\U000003A1": "P", "\U000003A4": "T", "\U000003A7": "X",
})
_HOMOGLIFOS_CHARS = "".join(chr(c) for c in HOMOGLIFOS)
PALABRA_MIXTA_RE = re.compile(
    rf"(?<!\w)(?=\w*[A-Za-zÁÉÍÓÚÑáéíóúñ])(?=\w*[{_HOMOGLIFOS_CHARS}])\w+(?!\w)"
)

# Símbolos tipográficos que se cambian por su equivalente simple 
SIMBOLOS_EQUIVALENTES = str.maketrans({
    "´": "'", "′": "'", "″": '"', "‐": "-", "‑": "-", "‒": "-", "―": "—",
    "•": " ", "●": " ", "▪": " ", "›": " ", "‹": " ", "·": " ", "§": " ",
    "©": " ", "®": " ", "™": " ", "¨": " ", "\U0000FFFD": " ",   # FFFD = carácter de reemplazo
})

# Frases pegadas
FRASE_PEGADA_RE = re.compile(r"(?<=[a-záéíóúñ]{2})([.!?:])(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])")
MINUSCULA_MAYUSCULAS_RE = re.compile(r"(?<=[a-záéíóúñ]{2})(?=[A-ZÁÉÍÓÚÑ]{2,})")
DIGITO_PALABRA_RE = re.compile(r"(?<=\d)(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])")

# Huellas de fuente (firmas, créditos de foto, pies de página)
# Cada una se documenta en el reporte
HUELLAS_FUENTE = {
    "firma_reuters": re.compile(r"^[^()]{0,80}\(Reuters\)\s*[-–—]\s*"),
    "reuters_en_medio": re.compile(r"\(Reuters\)\s*[-–—]?"),   # fake que copian un cable de Reuters
    "credito_foto_final": re.compile(r"(?i)(?:featured|feature)\s+(?:image|photo)(?!\w).*$", re.S),
    "credito_foto_parentesis": re.compile(r"(?i)\((?:photos?|images?)\s+(?:by|via|credit|from)[^)]*\)"),
    "pie_21wire": re.compile(r"(?i)(?:READ MORE [\w\s]{0,40}NEWS AT:|SUPPORT (?:OUR WORK|21WIRE)).*$", re.S),
    "parentesis_vacio": re.compile(r"\(\s*(?:graphic|photo|video|image)?\s*[:\-]?\s*\)", re.I),
}

NUMERO_RE = re.compile(r"(?<![A-Za-zÁÉÍÓÚÑáéíóúñ])\d+(?:[.,:]\d+)*")
MARCADOR_NUMERO = "*NUMBER*"
ESPACIOS_RE = re.compile(r"\s+")
ESPACIO_ANTES_PUNTUACION_RE = re.compile(r"\s+([.,;:!?)])")

# Funciones del pipeline
def _a_bytes_cp1252(fragmento: str) -> bytes:
    salida = b""
    for c in fragmento:
        try:
            salida += c.encode("cp1252")
        except UnicodeEncodeError:
            salida += c.encode("latin-1")
    return salida


def reparar_codificacion(texto: str) -> str:
    # Repara los dos tipos de mojibake y decodifica entidades HTML
    def _utf8(m: re.Match) -> str:
        try:
            return _a_bytes_cp1252(m.group(0)).decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return m.group(0)

    def _c1(m: re.Match) -> str:
        try:
            return bytes([ord(m.group(0))]).decode("cp1252")
        except UnicodeDecodeError:          # 0x81, 0x8D, 0x8F, 0x90, 0x9D no existen en cp1252
            return ""

    for roto, bueno in MOJIBAKE_RESTOS.items():
        texto = texto.replace(roto, bueno)
    texto = MOJIBAKE_UTF8_RE.sub(_utf8, texto)
    texto = C1_RE.sub(_c1, texto)
    texto = HTML_ENTIDAD_RE.sub(lambda m: html.unescape(m.group(0)), texto)
    return HTML_ENTIDAD_RE.sub(" ", texto)


def normalizar_unicode(texto: str) -> str:
    # NFC, espacios, y quita invisibles 
    texto = unicodedata.normalize("NFC", texto)
    salida = []
    for c in texto:
        categoria = unicodedata.category(c)
        if c.isspace():                            
            salida.append(" ")
        elif categoria in ("Cc", "Cf", "Co", "Cs"):  
            continue
        else:
            salida.append(c)
    return "".join(salida)


def reparar_homoglifos(texto: str, idioma: str) -> str:
    # Cambia letras cirílicas/griegas camufladas por la latina. En español, en todo el texto; en otros idiomas, solo dentro de palabras que mezclan alfabetos.
    if idioma == "es":
        return texto.translate(HOMOGLIFOS)
    return PALABRA_MIXTA_RE.sub(lambda m: m.group(0).translate(HOMOGLIFOS), texto)


def quitar_huellas_fuente(texto: str) -> str:
    for regex in HUELLAS_FUENTE.values():
        texto = regex.sub(" ", texto)
    return texto


def quitar_urls(texto: str) -> str:
    return URL_RE.sub(" ", texto)


def separar_frases_pegadas(texto: str) -> str:
    texto = FRASE_PEGADA_RE.sub(r"\1 ", texto)
    texto = re.sub(r"[@#]?\w+", lambda m: m.group(0) if m.group(0)[0] in "@#"   # no rompe @usuarios ni #hashtags
                   else MINUSCULA_MAYUSCULAS_RE.sub(" ", m.group(0)), texto)
    return DIGITO_PALABRA_RE.sub(" ", texto)


def quitar_emojis_y_simbolos(texto: str) -> str:
    return EMOJI_RE.sub(" ", texto).translate(SIMBOLOS_EQUIVALENTES)


def quitar_otros_alfabetos(texto: str) -> str:
    # Quita letras que no son latinas (hebreo, árabe, chino...). Conserva º, ª y µ.
    conservar = {"º", "ª", "µ"}
    return "".join(
        c for c in texto
        if not c.isalpha() or c in conservar or unicodedata.name(c, "").startswith("LATIN")
    )


def unificar_numeros(texto: str) -> str:
    return NUMERO_RE.sub(MARCADOR_NUMERO, texto)


def normalizar_espacios(texto: str) -> str:
    texto = ESPACIOS_RE.sub(" ", texto).strip()
    return ESPACIO_ANTES_PUNTUACION_RE.sub(r"\1", texto)


# Pipeline
pasos = [  # (nombre en params.yaml, función, recibe idioma)
    ("reparar_codificacion", reparar_codificacion, False),
    ("normalizar_unicode", normalizar_unicode, False),
    ("reparar_homoglifos", reparar_homoglifos, True),
    ("quitar_huellas_fuente", quitar_huellas_fuente, False),
    ("quitar_urls", quitar_urls, False),
    ("separar_frases_pegadas", separar_frases_pegadas, False),
    ("quitar_emojis_y_simbolos", quitar_emojis_y_simbolos, False),
    ("quitar_otros_alfabetos", quitar_otros_alfabetos, False),
    ("unificar_numeros", unificar_numeros, False),
    ("normalizar_espacios", normalizar_espacios, False),
]

def quitar_vacios_y_duplicados(df: pd.DataFrame, columna: str, min_caracteres: int) -> tuple[pd.DataFrame, dict]:
    # Quita vacíos (o muy cortos) y duplicados exactos
    prioridad = {"replica": 0, "modelo_b": 1, "generalizacion": 2}
    df = df.assign(_orden=df["uso"].map(prioridad).fillna(9)).sort_values(["_orden", "id"], kind="stable")
    cortos = df[columna].str.strip().str.len() < min_caracteres
    df = df[~cortos]
    duplicados = df.duplicated(subset=[columna], keep="first")
    conflictos = int(df[df[columna].duplicated(keep=False)].groupby(columna)["label"].nunique().gt(1).sum())
    df = df[~duplicados].drop(columns="_orden")
    return df, {"vacios_o_cortos": int(cortos.sum()), "duplicados": int(duplicados.sum()),
                "duplicados_con_label_distinto": conflictos}


def quitar_tretiakov_de_posadas_v2(df: pd.DataFrame, min_caracteres: int = 50) -> tuple[pd.DataFrame, int]:
    # Quita de Posadas v2 las noticias que contienen un texto de Tretiakov 
    tretiakov = [t for t in df.loc[df["fuente"] == "tretiakov", "texto_limpio"] if len(t) >= min_caracteres]
    v2 = df["fuente"] == "posadas_v2"
    contiene = df.loc[v2, "texto_limpio"].map(lambda t: any(x in t for x in tretiakov))
    quitar = contiene[contiene].index
    return df.drop(index=quitar), len(quitar)


def validar(df: pd.DataFrame) -> dict:
    # Comprobaciones de calidad. Cada valor debe ser 0 (o True)
    t = df["texto_limpio"]
    return {
        "emojis": int(t.str.count(EMOJI_RE.pattern).sum()),
        "controles_c1": int(t.str.count(C1_RE.pattern).sum()),
        "mojibake_utf8": int(t.str.count(MOJIBAKE_UTF8_RE.pattern).sum()),
        "homoglifos_es": int(t[df["idioma"] == "es"].str.count(f"[{_HOMOGLIFOS_CHARS}]").sum()),
        "urls": int(t.str.count(URL_RE.pattern).sum()),
        "entidades_html": int(t.str.count(HTML_ENTIDAD_RE.pattern).sum()),
        "vacios": int((t.str.len() == 0).sum()),
        "duplicados": int(t.duplicated().sum()),
        "labels_validos": bool(df["label"].isin([0, 1]).all()),
    }


def limpiar(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    reporte: dict = {"filas_entrada": len(df)}
    df = df.copy()
    df["texto"] = df["texto"].fillna("").astype(str)

    df, r = quitar_vacios_y_duplicados(df, "texto", 1)
    reporte["antes_de_limpiar"] = r

    df["texto_limpio"] = df["texto"]
    reporte["pasos"] = {}
    for nombre, funcion, usa_idioma in pasos:
        if not params["pasos"].get(nombre, True):
            reporte["pasos"][nombre] = "apagado"
            continue
        antes = df["texto_limpio"]
        if usa_idioma:
            despues = pd.Series([funcion(t, i) for t, i in zip(antes, df["idioma"])], index=df.index)
        else:
            despues = antes.map(funcion)
        cambiados = antes != despues
        reporte["pasos"][nombre] = {
            "docs_modificados": int(cambiados.sum()),
            "caracteres_netos_quitados": int(antes.str.len().sum() - despues.str.len().sum()),
            "docs_modificados_por_fuente_label": {
                f"{f}_{l}": int(n) for (f, l), n in cambiados.groupby([df["fuente"], df["label"]]).sum().items()
            },
        }
        df["texto_limpio"] = despues

    df, r = quitar_vacios_y_duplicados(df, "texto_limpio", params["min_caracteres"])
    reporte["despues_de_limpiar"] = r

    if params.get("quitar_tretiakov_de_posadas_v2", True):
        df, n = quitar_tretiakov_de_posadas_v2(df)
        reporte["tretiakov_quitadas_de_posadas_v2"] = n

    reporte["validacion"] = validar(df)
    reporte["filas_salida"] = len(df)
    reporte["filas_por_fuente_label"] = {
        f"{f}_{l}": int(n) for (f, l), n in df.groupby(["fuente", "label"]).size().items()
    }
    return df.reset_index(drop=True), reporte


def main(ruta_params: str = "params.yaml") -> None:
    params = yaml.safe_load(Path(ruta_params).read_text(encoding="utf-8"))["limpieza"]
    df = pd.read_csv(params["entrada"])
    limpio, reporte = limpiar(df, params)

    columnas = ["id", "fuente", "archivo", "idioma", "uso", "texto_limpio", "label"]
    if params.get("conservar_texto_original", False):
        columnas.insert(5, "texto")
    Path(params["salida"]).parent.mkdir(parents=True, exist_ok=True)
    limpio[columnas].to_csv(params["salida"], index=False, encoding="utf-8")

    Path(params["reporte"]).parent.mkdir(parents=True, exist_ok=True)
    Path(params["reporte"]).write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    v = reporte["validacion"]
    fallos = {k: x for k, x in v.items() if (x is not True and x != 0)}
    print(f"Filas: {reporte['filas_entrada']} -> {reporte['filas_salida']}")
    if fallos:
        raise SystemExit(f"Validación FALLÓ: {fallos}  (ver {params['reporte']})")
    print("Validación OK:", v)


if __name__ == "__main__":
    main()