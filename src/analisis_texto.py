"""
Análisis de texto:
- Identificación de emojis
- Caracteres y palabras extrañas
- Detonantes
- Contexto
- Ortografía
"""
# Importación de librerías y funciones
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer 

# Listas constantes
emoji_re = re.compile("[\U0001F300-\U0001FAFF]|[\U0001F1E6-\U0001F1FF]{2}")   # pictogramas + banderas (pares)
dingbat_re = re.compile("[\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF]")             # ✓ ♦ ☢: símbolos, no emojis
c1_re = re.compile("[\x80-\x9f]")              # mojibake Windows-1252 leído como Latin-1 (\x92 = ’)
homoglifo_re = re.compile(r"\b(?=\w*[A-Za-zÁÉÍÓÚÑáéíóúñ])(?=\w*[\u0370-\u03FF\u0400-\u04FF])\w+\b")  # latín + cirílico/griego
frase_pegada_re = re.compile(r"[a-záéíóúñ]{2,}[.!?:][A-ZÁÉÍÓÚÑ][a-záéíóúñ]+")      # "off.Watch" (salto de línea perdido)
patrones_tipicos = re.compile("(?:Ã[\u0080-¿]|â€[™œ\u009d“”¦\u0098\u0093\u0094]|Â[ -¿])")
url = re.compile(r"https?://\S+|www\.\S+|\b(?:pic\.twitter\.com|t\.co|bit\.ly|youtu\.be|goo\.gl|tmsnrt\.rs)/\S+")
html_entidad_re = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});")
palabra_re = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)

puntuacion_comun = set(".,;:!?'\"()[]{}-–—_/\\%$&@#*+=<>|~^`¿¡«»“”‘’…°")
caracteres_ignorados ={"\ufe0f","\u200d"}
negaciones = {"not", "no", "never", "nor", "nunca", "jamás", "tampoco", "ni", "sin"}  # "n't" se cuenta aparte


def nombre_caracter(c: str) -> str:
    """Nombre Unicode; para los controles C1 dice qué carácter de Windows-1252 eran."""
    if c1_re.fullmatch(c):
        try:
            return f"C1 (cp1252 → {bytes([ord(c)]).decode('cp1252')})"
        except UnicodeDecodeError:
            return "C1 (sin equivalente cp1252)"
    return unicodedata.name(c, "?")

# Clase para carga de datos
class Carga_Corpus:
    columnas_req = {"id", "fuente", "idioma", "texto", "label"}

    def __init__(self, ruta:str, quitar_vacios: bool = True, quitar_duplicados: bool = True):
        self.ruta = ruta
        self.quitar_vacios, self.quitar_duplicados = quitar_vacios, quitar_duplicados
        self.reporte: dict = {}

    def cargar(self) -> pd.DataFrame:
        df = pd.read_csv(self.ruta)
        faltantes = self.columnas_req - set(df.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas: {faltantes}")
        df["texto"] = df["texto"].fillna("").astype(str)
        self.reporte["filas_leidas"] = len(df)
        vacios = df["texto"].str.strip().eq("")
        self.reporte["vacios"] = int(vacios.sum())
        if self.quitar_vacios:
            df = df[~vacios]
        duplicados = df.duplicated(subset=["texto"], keep="first")
        self.reporte["duplicados"] = int(duplicados.sum())
        self.reporte["duplicados_label_distinto"] = int(df.groupby("texto")["label"].nunique().gt(1).sum())
        if self.quitar_duplicados:
            df = df[~duplicados]
        self.reporte["filas_finales"] = len(df)
        return df.reset_index(drop=True)

# Clase para inventario de caracteres
class Inventario_Caracteres:

    def __init__(self, textos:pd.Series):
        self.contador: Counter = Counter()
        for texto in textos:
            self.contador.update(texto)
        self._tabla: pd.DataFrame | None = None

    @staticmethod
    def clasificar(caracter:str) -> str:
        if caracter in caracteres_ignorados:
            return "componente_emoji"
        if emoji_re.match(caracter):
            return "emoji"
        if dingbat_re.match(caracter):
            return "simbolo_dingbat"
        if c1_re.match(caracter):
            return "mojibake_c1"
        categoria = unicodedata.category(caracter)
        if categoria[0] == "C" and caracter not in "\n\r\t":
            return "control_o_invisible"
        if caracter in "\n\r\t":
            return "salto_o_tab"
        if caracter.isspace():
            return "espacio" if caracter == " " else "espacio_raro"
        if caracter.isalpha():
            nombre = unicodedata.name(caracter, "")
            if nombre.startswith("LATIN"):
                return "letra_latina" if caracter.isascii() else "letra_latina_acentuada"
            return "letra_otro_alfabeto"
        if caracter.isdigit():
            return "digito"
        if caracter in puntuacion_comun:
            return "puntuacion_comun"
        return "simbolo_raro"

    def tabla(self) -> pd.DataFrame:
        if self._tabla is None:
            filas = [
                {
                    "caracter": repr(c)[1:-1],
                    "codepoint": f"U+{ord(c):04X}",
                    "nombre": nombre_caracter(c),
                    "tipo": self.clasificar(c),
                    "conteo": n,
                    "_car": c,
                }
                for c, n in self.contador.items()
            ]
            self._tabla = pd.DataFrame(filas). sort_values("conteo", ascending=False, ignore_index=True)
        return self._tabla

    def resumen_por_tipo(self) -> pd.DataFrame:
        t = self.tabla()
        salida = t.groupby("tipo").agg(distintos=("conteo", "size"), total=("conteo", "sum"))
        salida["pct_total"] = (salida["total"] / salida["total"].sum() * 100).round(4)
        return salida.sort_values("total", ascending = False)

    def por_tipo(self, tipo:str, top:int = 20) -> pd.DataFrame:
        t = self.tabla()
        return t[t["tipo"] == tipo].drop(columns="_car").head(top)

    def caracteres_raros(self) -> set[str]:
        t = self.tabla()
        tipos = {"simbolo_raro", "simbolo_dingbat", "mojibake_c1", "control_o_invisible", "espacio_raro",
                 "letra_otro_alfabeto"}
        return set(t.loc[t["tipo"].isin(tipos), "_car"])

# Perfil por documento para indicadores
class Perfilador_Textos:
    def __init__(self, caracteres_raros: set[str]):
        self.expreg_raros = (
            re.compile("[" + re.escape("".join(sorted(caracteres_raros))) + "]") if caracteres_raros else None
        )

    def perfilar(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df["texto"]
        salida = df[["id", "fuente", "idioma", "label"]].copy()   # "suente" → "fuente"
        salida["n_chars"] = s.str.len()
        salida["n_palabras"] = s.str.count(r"\S+")
        salida["n_emojis"] = s.str.count(emoji_re.pattern)
        salida["n_mojibake"] = s.str.count(patrones_tipicos.pattern)       # tipo UTF-8 leído como cp1252 (Ã©)
        salida["n_mojibake_c1"] = s.str.count(c1_re.pattern)               # tipo cp1252 leído como Latin-1 (\x92)
        salida["n_homoglifos"] = s.str.count(homoglifo_re.pattern)         # "cоntar" con о cirílica
        salida["n_frases_pegadas"] = s.str.count(frase_pegada_re.pattern)  # "off.Watch"
        salida["n_urls"] = s.str.count(url.pattern)                        # faltaba
        salida["n_entidades_html"] = s.str.count(html_entidad_re.pattern)  # faltaba
        salida["n_crlf"] = s.str.count(r"\r\n")
        salida["n_exclam"] = s.str.count("!")
        salida["n_raros"] = s.str.count(self.expreg_raros.pattern) if self.expreg_raros else 0
        salida["n_palabras_mayus"] = s.str.count(r"\b[A-ZÁÉÍÓÚÜÑ]{3,}\b")
        salida["pct_mayus"] = (
            salida["n_palabras_mayus"] / salida["n_palabras"].clip(lower=1) * 100
        ).round(2)
        return salida
    @staticmethod
    def prevalencia(perfil: pd.DataFrame, por: str = "label") -> pd.DataFrame:
        cols = ["n_emojis", "n_mojibake", "n_mojibake_c1", "n_homoglifos", "n_frases_pegadas",
                "n_urls", "n_entidades_html", "n_raros", "n_exclam"]
        pct = perfil.groupby(por)[cols].agg(lambda x: (x > 0).mean() * 100).add_prefix("pct_docs_")
        media = perfil.groupby(por)[["n_chars", "n_palabras", "pct_mayus"]].mean()
        return pd.concat([media, pct], axis = 1). round(2)

# Emoji
class Analizador_Emojis:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def top(self, n:int = 20) -> pd.DataFrame:
        contador = Counter()
        for texto in self.df["texto"]:
            contador.update(emoji_re.findall(texto))
        return pd.DataFrame(
            [(e, " + ".join(unicodedata.name(c, "?") for c in e), k) for e, k in contador.most_common(n)],
            columns=["emoji", "nombre", "conteo"],
        )
    def por_grupo(self, por: str = "fuente") -> pd.DataFrame:
        tiene = self.df["texto"].str.contains(emoji_re.pattern, regex=True)
        return (
            tiene.groupby(self.df[por])
            .agg(docs_con_emoji="sum", pct="mean")
            .assign(pct=lambda d: (d.pct * 100).round(3))
        )

# Palabras raras
class Analizador_Palabras_Raras:
    reglas = {
        "letra_repetida_3+": re.compile(r"\b\w*([^\W\d_])\1{2,}\w*\b"),           # solo letras: ya no cuenta 000
        "mezcla_letras_digitos": re.compile(r"\b(?=\w*[^\W\d_])(?=\w*\d)\w+\b"),
        "token_muy_largo_20+": re.compile(r"\b\w{20,}\b"),
        "mayuscula_intermedia": re.compile(r"\b\w*[a-záéíóúñ][A-ZÁÉÍÓÚÑ]\w*\b"),  # también FilesSUPPORT
        "frase_pegada": frase_pegada_re,
        "homoglifo": homoglifo_re,
        "url": url,
        "mojibake": patrones_tipicos,
        "mojibake_c1": c1_re,
        "entidad_html": html_entidad_re,
    }

    def __init__(self, df: pd.DataFrame, muestra: int | None = 10000, semilla: int = 42):
        self.df = df.sample(min(muestra, len(df)), random_state = semilla) if muestra else df

    def resumen(self) -> pd.DataFrame:
        filas = []
        for nombre, regex in self.reglas.items():
            aciertos = self.df["texto"].map(lambda t: len(regex.findall(t)))
            filas.append(
                {"regla": nombre, "pct_docs": round((aciertos > 0).mean() * 100, 2), "total": int(aciertos.sum())}
            )
        return pd.DataFrame(filas).set_index("regla")

    def ejemplos(self, regla:str, n: int = 15) -> list[str]:
        regex = self.reglas[regla]
        contador = Counter()
        for texto in self.df["texto"]:
            contador.update(m.group(0) for m in regex.finditer(texto))
        return [p for p, _ in contador.most_common(n)]

# Palabras detonantes
class Analizador_Detonantes:
    def __init__(self, min_df: int = 20, max_features: int = 30000):
        self.min_df, self.max_features = min_df, max_features

    def ajustar(self, df: pd.DataFrame, idioma: str, top: int = 2000,
                excluir_fuentes: tuple[str, ...] = ("posadas_v2",)) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = df[(df["idioma"] == idioma) & ~df["fuente"].isin(excluir_fuentes)]   # v2 = prueba de generalización
        vectorizador = CountVectorizer(
            binary = True, lowercase = True, min_df = self.min_df, max_features = self.max_features,
            stop_words=None, token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        )
        X = vectorizador.fit_transform(d["texto"]).tocsc()
        y = d["label"].to_numpy()
        n1, n0 = (y == 1).sum(), (y == 0).sum()
        c1 = np.asarray(X[y == 1].sum(0)).ravel()
        c0 = np.asarray(X[y == 0].sum(0)).ravel()
        p1, p0 = (c1 + 0.5) / (n1 + 1), (c0 + 0.5) / (n0 + 1)
        log_odds = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
        # Error estándar del log odds ratio (Woolf): z = cuántas desviaciones se aleja de 0
        varianza = 1 / (c1 + 0.5) + 1 / (n1 - c1 + 0.5) + 1 / (c0 + 0.5) + 1 / (n0 - c0 + 0.5)
        z = log_odds / np.sqrt(varianza)
        palabras = vectorizador.get_feature_names_out()

        fuentes = d["fuente"].to_numpy()
        conteo_fuente = pd.DataFrame({f: np.asarray(X[fuentes == f].sum(0)).ravel() for f in np.unique(fuentes)})
        dominante = conteo_fuente.idxmax(axis = 1)
        porcentaje = (conteo_fuente.max(axis = 1) / (c1 + c0).clip(min = 1) * 100).round(1)

        tabla = pd.DataFrame({
            "palabra": palabras, "docs_label0": c0, "docs_label1":c1, "log_odds": log_odds.round(3),
            "z": z.round(2), "fuente_dominante": dominante, "pct_fuente": porcentaje,
            "riesgo_fuga": porcentaje >= 90,
        })
        return (
            tabla.sort_values("z", ascending=False).head(top).reset_index(drop = True),
            tabla.sort_values("z").head(top).reset_index(drop = True),
        )

# Contexto
def palabra_en_contexto(textos: pd.Series, palabra: str, ancho:int = 70, n: int = 5, semilla: int = 42) -> list[str]:
    regex = re.compile(rf"\b{re.escape(palabra)}\b", re.IGNORECASE)
    encontrados = textos[textos.str.contains(regex, regex=True)]
    salida = []
    muestra = encontrados.sample(min(n, len(encontrados)), random_state=semilla) if len(encontrados) else []
    for texto in muestra:
        m = regex.search(texto)
        a, b = max(0, m.start() - ancho), min(len(texto), m.end() + ancho)
        salida.append(("…" if a else "") + re.sub(r"\s+", " ", texto[a:b]) + ("…" if b < len(texto) else ""))
    return salida


class Analizador_Impacto_Limpieza:
    def __init__(self, df: pd.DataFrame, muestra: int = 2500, semilla: int = 42):
        # muestra POR idioma: antes 5.000 mezclados (~290 en español)
        self.df = pd.concat([g.sample(min(muestra, len(g)), random_state=semilla)
                             for _, g in df.groupby("idioma")])
        self.estrategias = {
            "solo_ascii": lambda t: re.sub(r"[^\x00-\x7f]", "", t),
            "solo_alfanumerico": lambda t: re.sub(r"[^A-Za-z0-9\s]", "", t),
            "alfanumerico_unicode": lambda t: re.sub(r"[^\w\s]", "", t),   # conserva tildes y ñ
            "minusculas": lambda t: t.lower(),
        }

    @staticmethod
    def _negaciones(t: str) -> int:
        palabras = [p.lower() for p in re.findall(r"[\w'’]+", t)]
        return sum(p in negaciones or p.endswith(("n't", "n’t")) for p in palabras)

    @staticmethod
    def _comillas(t: str) -> int:
        return len(re.findall("[\"“”«»]", t))

    @staticmethod
    def _mayusculas(t: str) -> int:
        return len(re.findall(r"\b[A-ZÁÉÍÓÚÜÑ]{3,}\b", t))

    def ejecutar(self) -> pd.DataFrame:
        filas = []
        for idioma, textos in self.df.groupby("idioma")["texto"]:
            filas += self._medir(idioma, textos)
        return pd.DataFrame(filas).set_index(["idioma", "estrategia"])

    def _medir(self, idioma: str, textos: pd.Series) -> list[dict]:
        self.textos, filas = textos, []
        total = textos.str.len().sum()
        for nombre, funcion in self.estrategias.items():
            limpio = self.textos.map(funcion)
            filas.append({
                "idioma": idioma, "estrategia": nombre,
                "pct_chars_perdidos": round((1 - limpio.str.len().sum() / total) * 100, 2),
                "pct_docs_pierden_negaciones": round(
                    (self.textos.map(self._negaciones) > limpio.map(self._negaciones)).mean() * 100, 2),
                "pct_docs_pierden_comillas": round(
                    (self.textos.map(self._comillas) > limpio.map(self._comillas)).mean() * 100, 2),
                "pct_docs_pierden_MAYUSCULAS": round(
                    (self.textos.map(self._mayusculas) > limpio.map(self._mayusculas)).mean() * 100, 2),
                "pct_docs_pierden_tildes_ñ": round(
                    (textos.str.count("[áéíóúüñÁÉÍÓÚÜÑ]") > limpio.str.count("[áéíóúüñÁÉÍÓÚÜÑ]")).mean() * 100, 2),
            })
        return filas   

# Ortografía
class AnalizadorOrtografia:
    def __init__(self, muestra_por_idioma: int = 5000, semilla: int = 42):
            self.muestra, self.semilla = muestra_por_idioma, semilla
            try:
                import spellchecker  # noqa: F401
                self.tiene_corrector = True
            except ImportError:
                self.tiene_corrector = False
            try:
                import wordfreq  # noqa: F401
                self.tiene_wordfreq = True
            except ImportError:
                self.tiene_wordfreq = False
            self.umbral_zipf = 1.0   # zipf 1 ≈ una vez cada 100 millones de palabras
    
    @staticmethod
    def _vocabulario(textos: pd.Series) -> Counter:
        contador = Counter()
        for texto in textos:
            contador.update(
                p for p in palabra_re.findall(url.sub(" ", texto))          # sin URLs: "https" ya no sale
                if len(p) >= 3 and p.lower() == p and not re.search("['’]", p)
            )
        return contador

    def analizar(self, df: pd.DataFrame, idioma: str, top: int = 25) -> tuple[dict, pd.DataFrame]:
        d = df[df["idioma"] == idioma]
        d = d.sample(min(self.muestra, len(d)), random_state=self.semilla)
        vocabulario = self._vocabulario(d["texto"])
        if idioma == "es" and self.tiene_wordfreq:
            # el diccionario 'es' de pyspellchecker no trae "fue", "dijo", "años": usamos frecuencias de wordfreq
            from wordfreq import zipf_frequency
            desconocidas = {p for p in vocabulario if zipf_frequency(p, "es") < self.umbral_zipf}
            metodo = f"wordfreq (zipf < {self.umbral_zipf})"
        elif self.tiene_corrector:
            from spellchecker import SpellChecker
            desconocidas = SpellChecker(language=idioma).unknown(vocabulario)
            metodo = "pyspellchecker"
        else:
            desconocidas = {p for p, n in vocabulario.items() if n == 1}
            metodo = "hapax (respaldo)"
        tabla = pd.DataFrame(
            sorted(((p, vocabulario[p]) for p in desconocidas), key=lambda x: -x[1]),
            columns=["palabra", "frecuencia"],
        ).head(top)
        stats = {
            "idioma": idioma, "metodo": metodo, "docs_muestra": len(d),
            "vocabulario": len(vocabulario), "desconocidas": len(desconocidas),
            "pct_desconocidas": round(len(desconocidas) / max(len(vocabulario), 1) * 100, 2),
        }
        return stats, tabla

    def por_label(self, df: pd.DataFrame, idioma: str) -> pd.DataFrame:
        filas = []
        for etiqueta, grupo in df[df["idioma"] == idioma].groupby("label"):
            stats, _ = self.analizar(grupo, idioma, top=0)
            filas.append({"label": etiqueta, **stats})
        return pd.DataFrame(filas).set_index("label")