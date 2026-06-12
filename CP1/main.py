"""
main.py — Backend FastAPI para el sistema de recomendación de restaurantes de Madrid.

Arquitectura:
  · Motor NLP determinista (nlptown/bert + análisis local) → gratuito, sin API
  · RAG semántico (ChromaDB + Gemini embeddings) → fallback cuando NLP no ancla
  · Agente LangGraph → coordina ambos caminos y mantiene historial de conversación
  · Gemini Flash → solo se invoca cuando el NLP falla (≈ 0.009 €/consulta RAG)

Modos de uso:
  uvicorn main:app --reload              → servidor de desarrollo
  uvicorn main:app --host 0.0.0.0        → producción (Railway)

Primera ejecución (indexar base vectorial):
  python main.py --indexar

Archivos necesarios en el mismo directorio:
  analisis_restaurantes.csv   → generado por analizar_todos_restaurantes.py
  resenas_unificadas.csv      → reseñas originales (para indexar en ChromaDB)
  ranking.csv                 → ranking con Valoracion, Votaciones, Dirección
  .cache_coords.json          → caché de coordenadas

Variables de entorno:
  GEMINI_API_KEY              → para el fallback RAG (no obligatorio)
"""

import os
import re
import sys
import json
import math
import time
import unicodedata
import ast
import urllib.request as _ureq
from typing import List, Optional, TypedDict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_ANALISIS = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
ARCHIVO_RANKING  = os.path.join(BASE_DIR, "ranking.csv")
CACHE_COORDS    = os.path.join(BASE_DIR, ".cache_coords.json")

SOL_COORDS = (40.4168, -3.7038)  # Puerta del Sol — referencia de centro

# ═══════════════════════════════════════════════════════════════════════════════
# ESTADO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

df_global: Optional[pd.DataFrame] = None

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="API Recomendación Restaurantes Madrid",
    description="Sistema de recomendación basado en NLP local (nlptown/bert)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ═══════════════════════════════════════════════════════════════════════════════

class MensajeHistorial(BaseModel):
    role: str
    content: str


class ConsultaRequest(BaseModel):
    consulta: str
    historial: Optional[List[MensajeHistorial]] = []


class RecomendacionResponse(BaseModel):
    respuesta: str
    proyecto: str
    restaurantes: Optional[List[dict]] = []
    consulta_usuario: Optional[str] = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def _parsear_lista(val) -> list:
    """Convierte string de lista Python/JSON a lista real."""
    if isinstance(val, list):
        return val
    if not val or (isinstance(val, float) and math.isnan(val)):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        try:
            return json.loads(str(val))
        except Exception:
            return []


def _parsear_platos_frecuencia(top5_str: str) -> dict:
    """
    Convierte 'croquetas(34), pulpo(21)' → {'croquetas': 34, 'pulpo': 21}
    Compatible con el formato que espera el frontend (platos_frecuencia).
    """
    result = {}
    if not top5_str or not isinstance(top5_str, str):
        return result
    for parte in top5_str.split(","):
        parte = parte.strip()
        m = re.match(r'^(.+?)\((\d+)\)$', parte)
        if m:
            result[m.group(1).strip()] = int(m.group(2))
    return result


def _cargar_coords() -> dict:
    """Carga caché de coordenadas generada por generar_agente.py."""
    if os.path.exists(CACHE_COORDS):
        try:
            with open(CACHE_COORDS, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _cargar_dataframe() -> pd.DataFrame:
    """
    Carga analisis_restaurantes.csv (Proyecto B) + ranking.csv y combina.
    Aplica mapeo de columnas para que el frontend React funcione sin cambios.
    """
    print(f"Cargando {ARCHIVO_ANALISIS}...")
    df_nlp = pd.read_csv(ARCHIVO_ANALISIS)

    # Cargar ranking para Valoracion, Votaciones, Dirección
    df_rank = None
    if os.path.exists(ARCHIVO_RANKING):
        try:
            df_rank = pd.read_csv(ARCHIVO_RANKING, sep=";", skiprows=1)
            # Detectar columnas disponibles
            cols_rank = [c for c in ["Id_Restaurante", "Restaurante", "Votaciones", "Valoracion", "Dirección"]
                         if c in df_rank.columns]
            df_rank = df_rank[cols_rank]
            df_rank["Id_Restaurante"] = df_rank["Id_Restaurante"].astype(str)
            print(f"  ranking.csv cargado: {len(df_rank)} restaurantes")
        except Exception as e:
            print(f"  ⚠️  No se pudo cargar ranking.csv: {e}")
            df_rank = None

    # Normalizar id en df_nlp
    df_nlp["id_restaurante"] = df_nlp["id_restaurante"].astype(str)

    # Merge con ranking si está disponible
    if df_rank is not None:
        df_rank = df_rank.rename(columns={"Id_Restaurante": "id_restaurante"})
        df = pd.merge(df_nlp, df_rank, on="id_restaurante", how="left")
        # El nombre canónico: preferir ranking (más limpio), caer a columna 'nombre' del NLP
        if "Restaurante" in df.columns:
            df["nombre_display"] = df["Restaurante"].fillna(df["nombre"])
        else:
            df["nombre_display"] = df["nombre"]
        df["votaciones"] = pd.to_numeric(df.get("Votaciones", 0), errors="coerce").fillna(0).astype(int)
        df["valoracion_display"] = pd.to_numeric(df.get("Valoracion", df["valoracion_google"]), errors="coerce").fillna(df["valoracion_google"])
        df["direccion_display"] = df.get("Dirección", df.get("direccion", "")).fillna(df.get("direccion", "")).fillna("")
    else:
        df = df_nlp.copy()
        df["nombre_display"] = df["nombre"]
        df["votaciones"] = 0
        df["valoracion_display"] = df["valoracion_google"]
        df["direccion_display"] = df.get("direccion", "").fillna("")

    # Cargar coordenadas desde caché
    coords_cache = _cargar_coords()
    lats, lons = [], []
    for rid in df["id_restaurante"]:
        coords = coords_cache.get(str(rid))
        if coords and isinstance(coords, list) and len(coords) == 2:
            lats.append(coords[0])
            lons.append(coords[1])
        else:
            lats.append(None)
            lons.append(None)
    df["latitud"] = lats
    df["longitud"] = lons

    # Calcular distancia al centro (Sol)
    def _dist_sol(row):
        if pd.notna(row["latitud"]) and pd.notna(row["longitud"]):
            return _haversine(SOL_COORDS, (row["latitud"], row["longitud"]))
        return None

    df["dist_sol"] = df.apply(_dist_sol, axis=1)

    # Parsear columnas de criterios booleanos (pueden venir como string "True"/"False")
    for col in ["criterio_ninos", "criterio_mascotas", "criterio_terraza",
                "criterio_vistas", "criterio_musica_directo", "criterio_romantico",
                   "criterio_buen_postre", "criterio_precio_calidad",
                   "criterio_grupos_grandes", "criterio_vegano_vegetariano",
                   "criterio_sin_gluten"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: str(v).strip().lower() == "true" if not isinstance(v, bool) else v)
        else:
            df[col] = False

    # Parsear columna todos_platos si existe
    if "todos_platos" not in df.columns and "top5_platos" in df.columns:
        df["todos_platos"] = df["top5_platos"]

    print(f"  DataFrame listo: {len(df)} restaurantes")
    con_coords = df["latitud"].notna().sum()
    print(f"  Con coordenadas: {con_coords}/{len(df)}")
    return df


def _inicializar():
    global df_global
    df_global = _cargar_dataframe()
    print("Backend listo.")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES GEOGRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def _haversine(a: tuple, b: tuple) -> float:
    """Distancia en km entre dos puntos (lat, lon)."""
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


# Diccionario de zonas de Madrid — igual que en generar_agente.py
ZONAS_MADRID = {
    "sol": SOL_COORDS, "puerta del sol": SOL_COORDS,
    "gran via": (40.4200, -3.7040), "gran vía": (40.4200, -3.7040),
    "chueca": (40.4237, -3.6979),
    "malasana": (40.4260, -3.7060), "malasaña": (40.4260, -3.7060),
    "lavapies": (40.4078, -3.7018), "lavapiés": (40.4078, -3.7018),
    "la latina": (40.4110, -3.7092), "latina": (40.4110, -3.7092),
    "huertas": (40.4131, -3.6978), "letras": (40.4131, -3.6978),
    "embajadores": (40.4060, -3.7050),
    "tribunal": (40.4265, -3.6998),
    "alonso martinez": (40.4265, -3.6935), "alonso martínez": (40.4265, -3.6935),
    "colon": (40.4238, -3.6888), "colón": (40.4238, -3.6888),
    "recoletos": (40.4238, -3.6898),
    "retiro": (40.4153, -3.6844), "parque del retiro": (40.4153, -3.6844), "el retiro": (40.4153, -3.6844),
    "salamanca": (40.4298, -3.6831), "serrano": (40.4298, -3.6831),
    "goya": (40.4248, -3.6788), "jorge juan": (40.4248, -3.6748),
    "chamberi": (40.4350, -3.7000), "chamberí": (40.4350, -3.7000),
    "almagro": (40.4330, -3.6930),
    "arguelles": (40.4268, -3.7168), "argüelles": (40.4268, -3.7168),
    "moncloa": (40.4349, -3.7189),
    "principe pio": (40.4175, -3.7200), "príncipe pío": (40.4175, -3.7200),
    "opera": (40.4185, -3.7118), "ópera": (40.4185, -3.7118),
    "palacio real": (40.4179, -3.7143), "palacio": (40.4179, -3.7143),
    "tetuan": (40.4620, -3.6980), "tetuán": (40.4620, -3.6980),
    "cuatro caminos": (40.4456, -3.7010),
    "nuevos ministerios": (40.4488, -3.6922),
    "castellana": (40.4390, -3.6880),
    "azca": (40.4530, -3.6940),
    "hortaleza": (40.4780, -3.6540),
    "vallecas": (40.3878, -3.6580),
    "usera": (40.3878, -3.7118),
    "carabanchel": (40.3878, -3.7388),
    "plaza mayor": (40.4154, -3.7074),
    "san miguel": (40.4154, -3.7074),
    "prado": (40.4138, -3.6921), "museo del prado": (40.4138, -3.6921),
    "atocha": (40.4072, -3.6898),
    "chamartin": (40.4723, -3.6847), "chamartín": (40.4723, -3.6847),
    "fuencarral": (40.4265, -3.7018),
    "plaza espana": (40.4238, -3.7148), "plaza españa": (40.4238, -3.7148),
    "callao": (40.4208, -3.7058),
    "centro": SOL_COORDS, "centro madrid": SOL_COORDS,
    "bernabeu": (40.4531, -3.6884), "santiago bernabeu": (40.4531, -3.6884),
}


def _norm(s: str) -> str:
    """Normaliza string: minúsculas + quitar tildes."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _detectar_zona(consulta: str) -> Optional[tuple]:
    """Detecta si la consulta menciona una zona de Madrid y devuelve sus coordenadas."""
    c = _norm(consulta)
    # Patrones de extracción de zona
    patrones = [
        r'cerca de(?:l| la| los| las)?\s+([\w\s]+?)(?:\s+y|\s+quiero|\s+busco|\s+que|,|\.|$)',
        r'estoy en\s+([\w\s]+?)(?:\s+y|\s+quiero|\s+busco|,|\.|$)',
        r'en\s+([\w\s]{3,30}?)(?:\s+quiero|\s+busco|\s+que|,|\.|$)',
        r'por\s+([\w\s]{3,20}?)(?:\s+quiero|\s+busco|,|\.|$)',
        r'barrio de\s+([\w\s]+?)(?:,|\.|$)',
        r'zona de\s+([\w\s]+?)(?:,|\.|$)',
    ]
    for patron in patrones:
        m = re.search(patron, c)
        if m:
            zona_raw = m.group(1).strip()
            # Buscar en diccionario
            if zona_raw in ZONAS_MADRID:
                return ZONAS_MADRID[zona_raw]
            # Búsqueda parcial
            for nombre, coords in ZONAS_MADRID.items():
                if zona_raw in nombre or nombre in zona_raw:
                    return coords
    # Búsqueda directa en texto
    for nombre, coords in sorted(ZONAS_MADRID.items(), key=lambda x: -len(x[0])):
        if nombre in c:
            return coords
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE BÚSQUEDA 
# ═══════════════════════════════════════════════════════════════════════════════

# Cocinas y sus platos representativos (igual que generar_agente.py)
# Platos EXCLUSIVOS por cocina — solo los que identifican claramente el restaurante
# Se evitan platos genéricos que aparecen en cualquier restaurante (ceviche, curry, pasta, etc.)
COCINAS = {
    # Platos muy específicos de cocina mexicana auténtica
    "mexicana":    ["burrito", "quesadilla", "fajitas", "guacamole", "enchilada", "pozole",
                    "carnitas", "mole", "tacos cochinita", "tacos pastor", "chilaquiles",
                    "chile relleno", "tamales", "tostadas", "tlayudas"],

    # Italiana: pasta y pizza son genéricas, priorizar platos concretos
    "italiana":    ["carbonara", "cacio e pepe", "amatriciana", "lasana", "lasaña", "ossobuco",
                    "tiramisu", "panna cotta", "bruschetta", "focaccia", "risotto funghi",
                    "tagliatelle ragu", "pappardelle", "gnocchi", "cannoli", "ribollita",
                    "arancini", "carpaccio", "burrata", "caprese", "pizza napolitana",
                    "pizza margherita", "stracciatella", "saltimbocca"],

    # Japonesa: sushi es genérico en muchos restaurantes fusión
    "japonesa":    ["ramen", "gyozas", "edamame", "udon", "yakitori", "mochi", "katsu",
                    "takoyaki", "tonkatsu", "tempura ebi", "nigiri", "onigiri",
                    "okonomiyaki", "miso ramen", "soba"],

    # India: curry es muy genérico
    "india":       ["tikka masala", "biryani", "naan", "samosa", "korma", "dal",
                    "tandoori", "chapati", "pakora", "butter chicken", "palak paneer",
                    "chana masala", "lassi", "dosa", "saag"],

    # Peruana: solo platos genuinamente peruanos
    "peruana":     ["lomo saltado", "lomo salteado", "causa limena",
                    "anticuchos", "arroz chaufa", "aji de gallina", "ceviche amazonico",
                    "ceviche pacha", "pachamanquero", "suspiro limeno",
                    "leche de tigre", "chicharron peruano"],

    # Española clásica
    "española":    ["paella", "cocido madrileno", "fabada", "gazpacho", "salmorejo",
                    "patatas bravas", "croquetas jamon", "tortilla española", "pisto manchego",
                    "rabo de toro", "chuleton", "chuletón", "callos madrilenos", "oreja",
                    "cocido", "callos", "morcilla", "lechazo", "cordero asado"],

    # Asturiana: platos muy específicos
    "asturiana":   ["cachopo", "fabada asturiana", "oricios", "pote asturiano",
                    "merluza asturiana", "sidra", "cabrales", "casadielles"],

    # Gallega: platos muy específicos + variantes sin adjetivo
    "gallega":     ["pulpo gallego", "pulpo feria", "pulpo", "empanada gallega", "empanada",
                    "caldo gallego", "lacón con grelos", "lacon", "percebes", "zamburinas",
                    "berberechos", "navajas", "vieiras", "pimientos padron", "padron",
                    "filloas", "tarta de santiago", "queso tetilla", "tetilla",
                    "pote gallego", "zorza", "grelos", "ribeiro", "albarino"],

    # Vasca: platos muy específicos
    "vasca":       ["pintxos", "pintxo", "gilda", "bacalao pil pil", "txangurro",
                    "marmitako", "kokotxas", "bacalao al pil pil", "txakoli",
                    "merluza en salsa verde", "chipirones en su tinta"],

    # Francesa
    "francesa":    ["foie gras", "confit de pato", "ratatouille", "bouillabaisse",
                    "coq au vin", "escargots", "crepe suzette", "soufflé",
                    "cassoulet", "tartare", "magret"],

    # Griega
    "griega":      ["gyros", "souvlaki", "moussaka", "spanakopita", "tzatziki",
                    "baklava", "dolmades", "kleftiko", "taramasalata"],

    # Árabe/Libanesa
    "arabe":       ["shawarma", "falafel", "tabule", "hummus casero", "pita",
                    "baba ganoush", "labneh", "shakshuka", "kebab",
                    "kibbeh", "fattoush", "couscous"],

    # Venezolana
    "venezolana":  ["arepa", "arepas", "pabellon criollo", "cachapa", "cachapas",
                    "hallaca", "pernil", "caraotas", "tequeño", "tequeños",
                    "mandocas", "chicha", "pabellon"],

    # Colombiana
    "colombiana":  ["bandeja paisa", "ajiaco", "sancocho", "empanada colombiana",
                    "changua", "lechona", "tamales colombianos"],

    # China
    "china":       ["dim sum", "wonton", "pato pekin", "chow mein",
                    "spring roll", "dumplings", "baozi", "mapo tofu",
                    "pato laqueado", "cerdo agridulce"],

    # Tailandesa
    "tailandesa":  ["pad thai", "tom yum", "massaman", "curry verde thai",
                    "curry rojo thai", "satay", "som tam", "khao pad",
                    "larb", "mango sticky rice"],

    # Americana
    "americana":   ["smash burger", "pulled pork", "costillas bbq", "costillar",
                    "mac and cheese", "chicken wings", "brisket",
                    "coleslaw", "corn dog", "brownie"],

    # Mediterránea fusión
    "mediterranea": ["shakshuka", "baba ganoush", "labneh", "fattoush",
                     "couscous marroqui", "tajine", "merguez", "harira"],
}
# Nota: COCINAS se mantiene para compatibilidad futura pero la lógica de scoring
# usa los diccionarios internos de _score_cocina (PLATOS_PROPIOS / PLATOS_COMUNES)

SINONIMOS_COCINA = {
    "mexicano": "mexicana", "mexicana": "mexicana", "mexico": "mexicana", "mejico": "mexicana",
    "italiano": "italiana", "italiana": "italiana", "italia": "italiana",
    "japones": "japonesa", "japonesa": "japonesa", "japon": "japonesa", "sushi": "japonesa",
    "indio": "india", "india": "india", "hindu": "india",
    "peruano": "peruana", "peruana": "peruana", "peru": "peruana",
    "espanol": "española", "espanola": "española", "espana": "española", "asador": "española", "asadores": "española", "castellano": "española", "castellana": "española", "taberna": "española", "meson": "española", "mesón": "española",
    "asturiano": "asturiana", "asturiana": "asturiana", "asturias": "asturiana",
    "gallego": "gallega", "gallega": "gallega", "galicia": "gallega", "galego": "gallega", "galega": "gallega", "galeg": "gallega",
    "vasco": "vasca", "vasca": "vasca", "pais vasco": "vasca", "euskadi": "vasca",
    "frances": "francesa", "francesa": "francesa", "francia": "francesa",
    "griego": "griega", "griega": "griega", "grecia": "griega",
    "arabe": "arabe", "libanes": "arabe", "libano": "arabe", "marroqui": "arabe",
    "venezolano": "venezolana", "venezolana": "venezolana", "venezuela": "venezolana",
    "colombiano": "colombiana", "colombiana": "colombiana", "colombia": "colombiana",
    "chino": "china", "china": "china",
    "tailandes": "tailandesa", "tailandesa": "tailandesa", "tailandia": "tailandesa", "thai": "tailandesa",
    "americano": "americana", "americana": "americana", "usa": "americana",
    "mediterraneo": "mediterranea", "mediterranea": "mediterranea",
}

FRASES_COCINA = {
    "cocina gallega": "gallega", "comida gallega": "gallega", "cocina galega": "gallega", "comida galega": "gallega",
    "restaurante gallego": "gallega", "gastronomia gallega": "gallega",
    "cocina vasca": "vasca", "comida vasca": "vasca",
    "restaurante vasco": "vasca", "gastronomia vasca": "vasca",
    "cocina italiana": "italiana", "comida italiana": "italiana",
    "restaurante italiano": "italiana",
    "cocina japonesa": "japonesa", "comida japonesa": "japonesa",
    "restaurante japones": "japonesa",
    "cocina mexicana": "mexicana", "comida mexicana": "mexicana",
    "restaurante mexicano": "mexicana",
    "cocina peruana": "peruana", "comida peruana": "peruana",
    "restaurante peruano": "peruana",
    "cocina francesa": "francesa", "comida francesa": "francesa",
    "cocina china": "china", "comida china": "china",
    "cocina india": "india", "comida india": "india",
    "cocina americana": "americana", "comida americana": "americana",
    "cocina griega": "griega", "comida griega": "griega",
    "cocina arabe": "arabe", "comida arabe": "arabe",
    "cocina asturiana": "asturiana", "comida asturiana": "asturiana",
}

# Mapa de intenciones → criterio (igual que generar_agente.py)
INTENCIONES_CRITERIO = {
    "romantico": ["pareja", "romantico", "romantica", "intimo", "cena romantica",
                  "aniversario", "primera cita", "san valentin", "pedida",
                  "para dos", "velada", "noche especial", "cena especial"],
    "ninos": ["ninos", "ninas", "bebe", "bebes", "familia", "familiar",
              "peques", "pequenos", "crio", "crios", "chaval", "silla de bebe",
              "con los ninos", "plan familiar", "salir en familia"],
    "terraza": ["terraza", "terrazas", "exterior", "al aire libre", "fuera",
                "patio", "veladores", "jardin", "comer fuera"],
    "vistas": ["vistas", "vista", "panoramica", "azotea", "rooftop", "mirador", "skyline"],
    "musica_directo": ["musica en directo", "musica directo", "concierto",
                       "actuacion", "banda", "en vivo", "jazz", "flamenco"],
    "tranquilo": ["tranquilo", "tranquila", "silencioso", "sin ruido", "reposado", "relajado"],
    "precio_ok": ["economico", "barato", "precio", "relacion calidad", "asequible", "no muy caro"],
    "muy_valorado": ["mejor valorado", "mas valorado", "top", "el mejor", "altamente recomendado"],
}


def _detectar_cocina(consulta: str) -> Optional[str]:
    c = _norm(consulta)

    # 1. Frases completas primero
    for frase, cocina in FRASES_COCINA.items():
        if frase in c:
            return cocina

    # 2. Tokens individuales
    padded = " " + c + " "
    for sinonimo, cocina in SINONIMOS_COCINA.items():
        if (f" {sinonimo} " in padded
                or c == sinonimo
                or c.startswith(sinonimo + " ")
                or c.endswith(" " + sinonimo)):
            return cocina

    # 3. Prefijos para plurales y derivados
    tokens = c.split()
    for token in tokens:
        for sinonimo, cocina in SINONIMOS_COCINA.items():
            if len(sinonimo) >= 5 and token.startswith(sinonimo):
                return cocina

    return None
def _detectar_criterios(consulta: str) -> list:
    """Detecta criterios implícitos en la consulta (niños, terraza, etc.)."""
    c = _norm(consulta)
    encontrados = []
    for criterio, palabras in INTENCIONES_CRITERIO.items():
        if any(p in c for p in palabras):
            encontrados.append(criterio)
    return encontrados


def _parsear_platos_str(platos_str: str) -> list:
    """Convierte 'croquetas(34), pulpo(21)' → [{'nombre': 'croquetas', 'menciones': 34}, ...]"""
    if not platos_str or not isinstance(platos_str, str):
        return []
    result = []
    for parte in platos_str.split(","):
        parte = parte.strip()
        m = re.match(r'^(.+?)\((\d+)\)$', parte)
        if m:
            result.append({"nombre": m.group(1).strip(), "menciones": int(m.group(2))})
        elif parte:
            result.append({"nombre": parte, "menciones": 1})
    return result


def _score_cocina(row: pd.Series, cocina: str) -> float:
    """
    Score de afinidad con una cocina.

    Lógica de dos capas para evitar falsos positivos:

    1. Platos PROPIOS: muy identificativos de esa cocina (zamburiñas → gallega,
       ramen → japonesa, tikka masala → india...). Cada uno suma fuerte.

    2. Platos COMUNES: presentes en esa cocina pero también en muchas otras
       (pulpo, empanada, pasta...). Solo suman si hay al menos 1 plato propio
       o un nombre de restaurante que evoque la cocina.

    Umbral de entrada: el restaurante necesita al menos MIN_SEÑALES_PROPIAS
    señales de platos propios, O tener nombre_bonus + al menos 1 plato
    (propio o común). Sin eso, score = 0.
    """
    # Para gallega y vasca exigimos más señales propias (más fácil tener 1 plato de marisco suelto)
    MIN_SEÑALES_PROPIAS = 3 if cocina in ("gallega", "vasca") else 2

    # Platos tan exclusivos que con 1 mención ya son señal válida
    PLATOS_HIPERID = {
        "gallega": {"percebes", "vieiras", "filloas", "tetilla", "pote gallego",
                    "zorza", "caldo gallego", "lacon", "grelos", "albarino", "ribeiro"},
        "vasca":   {"pintxos", "pintxo", "gilda", "txangurro", "kokotxas",
                    "marmitako", "txakoli"},
    }

    # Platos propios (muy identificativos) y comunes (necesitan contexto)
    PLATOS_PROPIOS = {
        "gallega":     ["zamburinas", "zamburiñas", "percebes", "navajas", "vieiras",
                        "berberechos", "caldo gallego", "lacon", "grelos", "padron",
                        "filloas", "tetilla", "pote gallego", "zorza", "ribeiro", "albarino"],
        "vasca":       ["pintxos", "pintxo", "gilda", "bacalao pil pil", "txangurro",
                        "marmitako", "kokotxas", "txakoli", "chipirones en su tinta",
                        "bacalao al pil pil", "merluza en salsa verde", "pil pil"],
        "asturiana":   ["cachopo", "fabada asturiana", "oricios", "pote asturiano",
                        "cabrales", "casadielles", "sidra"],
        "japonesa":    ["ramen", "gyozas", "edamame", "udon", "yakitori", "mochi", "katsu",
                        "takoyaki", "tonkatsu", "nigiri", "onigiri", "okonomiyaki",
                        "miso ramen", "soba", "sashimi", "tempura ebi"],
        "india":       ["tikka masala", "biryani", "naan", "samosa", "korma", "dal",
                        "tandoori", "chapati", "pakora", "butter chicken", "palak paneer",
                        "chana masala", "lassi", "dosa", "saag"],
        "peruana":     ["lomo saltado", "lomo salteado", "causa limena", "anticuchos",
                        "arroz chaufa", "aji de gallina", "leche de tigre",
                        "ceviche amazonico", "ceviche pacha", "pachamanquero",
                        "suspiro limeno", "chicharron peruano", "tiradito"],
        "venezolana":  ["arepas", "arepa", "pabellon criollo", "cachapa", "cachapas",
                        "hallaca", "pernil", "caraotas", "tequeños", "mandocas", "chicha"],        "mexicana":    ["burrito", "quesadilla", "fajitas", "enchilada", "pozole",
                        "carnitas", "mole", "tacos cochinita", "tacos pastor", "chilaquiles",
                        "chile relleno", "tamales", "tostadas", "tlayudas"],
        "griega":      ["gyros", "souvlaki", "moussaka", "spanakopita", "tzatziki",
                        "baklava", "dolmades", "kleftiko", "taramasalata"],
        "italiana":    ["carbonara", "cacio e pepe", "amatriciana", "ossobuco",
                        "panna cotta", "risotto funghi", "tagliatelle ragu", "pappardelle",
                        "gnocchi", "cannoli", "ribollita", "arancini", "burrata",
                        "stracciatella", "saltimbocca", "pizza napolitana"],
        "francesa":    ["foie gras", "confit de pato", "bouillabaisse", "coq au vin",
                        "escargots", "crepe suzette", "soufflé", "cassoulet", "magret"],
        "arabe":       ["shawarma", "falafel", "tabule", "baba ganoush", "labneh",
                        "shakshuka", "kibbeh", "fattoush", "couscous", "pita"],
        "colombiana":  ["bandeja paisa", "ajiaco", "sancocho", "changua", "lechona",
                        "tamales colombianos"],
        "china":       ["dim sum", "wonton", "pato pekin", "chow mein", "baozi",
                        "mapo tofu", "pato laqueado"],
        "tailandesa":  ["pad thai", "tom yum", "massaman", "curry verde thai",
                        "curry rojo thai", "satay", "som tam", "larb", "mango sticky rice"],
        "americana":   ["smash burger", "pulled pork", "costillas bbq", "mac and cheese",
                        "chicken wings", "brisket", "coleslaw", "corn dog"],
        "española":    ["cocido madrileno", "fabada", "pisto manchego", "croquetas jamon",
                        "tortilla española", "rabo de toro", "callos madrilenos", "oreja",
                        "patatas bravas", "gazpacho", "salmorejo", "chuleton", "chuletón",
                        "cocido", "callos", "morcilla", "lechazo", "cordero asado"],
        "mediterranea": ["shakshuka", "baba ganoush", "labneh", "fattoush",
                         "couscous marroqui", "tajine", "merguez", "harira"],
    }
    # Platos que aparecen en múltiples cocinas — solo cuentan con contexto
    PLATOS_COMUNES = {
        "gallega":    ["pulpo", "empanada", "berberechos", "mejillones", "almejas"],
        "vasca":      ["bacalao", "merluza", "anchoas"],
        "italiana":   ["lasana", "lasaña", "bruschetta", "focaccia", "tiramisu",
                       "risotto", "pasta", "pizza"],
        "japonesa":   ["sushi", "tempura", "gyozas"],
        "peruana":    ["ceviche"],
        "mexicana":   ["tacos", "guacamole", "nachos"],
        "española":   ["paella", "jamón", "chorizo"],
        "francesa":   ["ratatouille", "tartare", "foie", "crepe"],
        "arabe":      ["hummus", "kebab"],
        "venezolana": ["pabellon", "chicha"],
        "americana":  ["brownie", "costillar"],
    }

    propios_def = PLATOS_PROPIOS.get(cocina, [])
    comunes_def = PLATOS_COMUNES.get(cocina, [])

    fuente = _norm(
        str(row.get("todos_platos", "") or "") + " " +
        str(row.get("terminos_tfidf", "") or "") + " " +
        str(row.get("nombre_display", "") or "") + " " +
        str(row.get("nombre", "") or "")
    )
    platos_lista = _parsear_platos_str(str(row.get("todos_platos", "") or ""))

    # Bonus por nombre del restaurante (señal muy fuerte)
    NOMBRES_COCINA = {
        "peruana":    ["tampu", "kausa", "peru", "lima", "inca"],
        "mexicana":   ["mexico", "mexicana", "taco", "azteca"],
        "japonesa":   ["japon", "sushi", "ramen", "tokyo", "osaka", "sibuya", "nikkei"],
        "italiana":   ["italia", "trattoria", "osteria", "pizzeria", "pasta"],
        "vasca":      ["euskal", "pintxo", "bilbao", "donosti", "txoko"],
        "gallega":    ["galicia", "galleg", "santiag", "marisqueria", "marisquer"],
        "asturiana":  ["astur", "asturias", "sidrer"],
        "griega":     ["grecia", "greek", "atenas", "hellas"],
        "arabe":      ["lebanese", "arab", "libano", "siria", "halal"],
        "venezolana": ["venezuela", "venezolan", "arepa"],
        "colombiana": ["colombia", "colombian"],
        "china":      ["china", "canton", "pekin", "shangai", "dragon"],
        "tailandesa": ["thai", "tailand", "bangkok", "siam"],
        "americana":  ["burger", "bbq", "smokehouse", "diner"],
        "española":   ["taberna", "meson", "mesón", "bodega", "tasca"],
    }
    nombre_norm = _norm(str(row.get("nombre_display", "") or ""))
    nombre_bonus = 0.0
    for hint in NOMBRES_COCINA.get(cocina, []):
        if hint in nombre_norm:
            nombre_bonus = 5.0
            break

    # Bonus fuerte si cocina_detectada del CSV coincide con la cocina buscada
    # Esto respeta el trabajo ya hecho por analizar_todos_restaurantes.py
    cocina_detectada_csv = _norm(str(row.get("cocina_detectada", "") or ""))
    cocina_ascii = _norm(cocina)
    if cocina_detectada_csv and (cocina_ascii in cocina_detectada_csv or cocina_detectada_csv in cocina_ascii):
        nombre_bonus = max(nombre_bonus, 10.0)

    # Bonus por columna tipo_cocina si existe en el CSV
    tipo = _norm(str(row.get("tipo_cocina", "") or row.get("cocina", "") or ""))
    if tipo and (cocina_ascii in tipo or tipo in cocina_ascii):
        nombre_bonus = max(nombre_bonus, 8.0)

    # Contar platos propios encontrados
    # Para gallega/vasca, platos genéricos de marisco (zamburiñas) necesitan ≥ 2 menciones.
    # Platos hiper-identificativos (percebes, vieiras, filloas...) valen con 1 mención.
    hiperid = PLATOS_HIPERID.get(cocina, set())
    MIN_MENCIONES_PROPIO = 2 if cocina in ("gallega", "vasca") else 1
    señales_propias = 0
    score_propios = 0.0
    for plato in propios_def:
        plato_norm = _norm(plato)
        if plato_norm in fuente:
            po = next(
                (p for p in platos_lista
                 if plato_norm in _norm(p["nombre"]) or _norm(p["nombre"]) in plato_norm),
                None
            )
            menciones = po["menciones"] if po else 1
            min_men = 1 if plato in hiperid else MIN_MENCIONES_PROPIO
            if menciones < min_men:
                continue   # señal demasiado débil
            señales_propias += 1
            score_propios += 3.0
            if menciones > 1:
                score_propios += math.log2(menciones)

    # Platos comunes: solo suman si hay contexto (platos propios O nombre_bonus)
    score_comunes = 0.0
    tiene_contexto = señales_propias >= MIN_SEÑALES_PROPIAS or nombre_bonus > 0
    if tiene_contexto:
        for plato in comunes_def:
            plato_norm = _norm(plato)
            if plato_norm in fuente:
                score_comunes += 1.5
                po = next(
                    (p for p in platos_lista
                     if plato_norm in _norm(p["nombre"]) or _norm(p["nombre"]) in plato_norm),
                    None
                )
                if po and po["menciones"] > 1:
                    score_comunes += math.log2(po["menciones"]) * 0.5

    # Sin contexto y sin nombre_bonus → score 0 (evita falsos positivos)
    if not tiene_contexto:
        return 0.0

    return round(score_propios + score_comunes + nombre_bonus, 2)


def _score_texto(row: pd.Series, tokens_query: list) -> float:
    """Score de coincidencia de tokens de consulta contra platos/términos del restaurante."""
    fuente = _norm(
        str(row.get("todos_platos", "") or "") + " " +
        str(row.get("terminos_tfidf", "") or "") + " " +
        str(row.get("nombre_display", "") or "")
    )
    score = 0.0
    for qt in tokens_query:
        if len(qt) < 3:
            continue
        if qt in fuente:
            score += 2.0
            # Bonus por menciones
            platos_lista = _parsear_platos_str(str(row.get("todos_platos", "") or ""))
            po = next((p for p in platos_lista if qt in _norm(p["nombre"]) or _norm(p["nombre"]) in qt), None)
            if po and po["menciones"] > 1:
                score += math.log2(po["menciones"])
    return score


def _score_calidad(row: pd.Series) -> float:
    """
    Score de calidad combinado 0-10 — idéntico a calcular_score_ranking del Proyecto A
    pero usando columnas del Proyecto B.

    Pesos:
      35% — valoración Google (0-5 → 0-10)
      35% — % reseñas positivas (nlptown)
      30% — avg_estrellas_modelo (nlptown, 1-5 → 0-10)
    """
    val = float(row.get("valoracion_display", 0) or 0)
    val_norm = (val / 5.0) * 10

    pct_pos = float(row.get("pct_positivo", 0) or 0)  # ya es 0-100

    avg_est = float(row.get("avg_estrellas_modelo", 3) or 3)
    avg_norm = ((avg_est - 1) / 4.0) * 10  # 1-5 → 0-10

    return round(val_norm * 0.35 + pct_pos / 10.0 * 0.35 + avg_norm * 0.30, 2)


def _pasa_criterio(row: pd.Series, criterio: str) -> bool:
    """Evalúa si un restaurante cumple un criterio de filtro."""
    # Criterios derivados de columnas NLP directas
    MAPA_CRITERIO_COL = {
        "ninos":               "criterio_ninos",
        "mascotas":            "criterio_mascotas",
        "terraza":             "criterio_terraza",
        "vistas":              "criterio_vistas",
        "musica_directo":      "criterio_musica_directo",
        "romantico":           "criterio_romantico",
        "buen_postre":         "criterio_buen_postre",
        "precio_calidad":      "criterio_precio_calidad",
        "grupos_grandes":      "criterio_grupos_grandes",
        "vegano_vegetariano":  "criterio_vegano_vegetariano",
        "sin_gluten":          "criterio_sin_gluten",
    }
    if criterio in MAPA_CRITERIO_COL:
        val = row.get(MAPA_CRITERIO_COL[criterio], False)
        if isinstance(val, str):
            val = val.strip().lower() == "true"
        return bool(val)

    # Criterios derivados de dimensiones numéricas
    if criterio == "tranquilo":
        ruido_neg = float(row.get("ruido_neg", 0) or 0)
        ruido_avg = float(row.get("ruido_avg_stars", 0) or 0)
        return ruido_neg == 0 or ruido_avg >= 4.0
    if criterio == "precio_ok":
        precio_avg = float(row.get("precio_avg_stars", 0) or 0)
        return precio_avg >= 4.0 if precio_avg > 0 else False
    if criterio == "muy_valorado":
        pct = float(row.get("pct_positivo", 0) or 0)
        val = float(row.get("valoracion_display", 0) or 0)
        return pct >= 90 and val >= 4.7
    if criterio == "centrico":
        dist = row.get("dist_sol")
        return dist is not None and float(dist) <= 1.5
    return True


def _fila_a_restaurante(row: pd.Series, distancia_km: Optional[float] = None) -> dict:
    """
    Convierte una fila del DataFrame al formato exacto que espera el frontend React.
    Mapea columnas del Proyecto B → shape del Proyecto A.
    """
    # Platos destacados: convertir "croquetas(34), pulpo(21)" → ["croquetas", "pulpo"]
    platos_lista = _parsear_platos_str(str(row.get("todos_platos", "") or row.get("top5_platos", "") or ""))

    # Filtrar nombres del personal para que no aparezcan como platos
    personal_str = str(row.get("personal_destacado", "") or "")
    nombres_personal = set()
    for parte in personal_str.split(","):
        nombre = re.sub(r'\(\d+\)$', '', parte).strip().lower()
        if nombre and nombre != "nan":
            nombres_personal.add(nombre)
            # También añadir partes individuales del nombre (ej: "yenifer adrián" → {"yenifer", "adrián"})
            for token in nombre.split():
                if len(token) > 3:
                    nombres_personal.add(token)

    platos_lista = [
        p for p in platos_lista
        if p["nombre"].lower() not in nombres_personal
        and not any(tok in nombres_personal for tok in p["nombre"].lower().split())
    ]
    platos_destacados = [p["nombre"] for p in platos_lista]

    # Platos frecuencia: {"croquetas": 34, "pulpo": 21}
    platos_frecuencia = {p["nombre"]: p["menciones"] for p in platos_lista}

    # Generar resumen automático si no hay columna 'resumen'
    resumen = str(row.get("resumen", "") or "")
    if not resumen or resumen == "nan":
        pct = float(row.get("pct_positivo", 0) or 0)
        n = int(row.get("n_resenas", 0) or 0)
        platos_top = platos_destacados[:3]
        partes = []
        if n > 0:
            partes.append(f"{n} reseñas analizadas con {pct:.0f}% de valoraciones positivas")
        if platos_top:
            partes.append(f"Platos más mencionados: {', '.join(platos_top)}")
        resumen = ". ".join(partes) if partes else ""

    # Mapear criterios de Proyecto B → campos booleanos que espera el React
    criterio_ninos             = bool(row.get("criterio_ninos", False))
    criterio_mascotas          = bool(row.get("criterio_mascotas", False))
    criterio_terraza           = bool(row.get("criterio_terraza", False))
    criterio_romantico         = bool(row.get("criterio_romantico", False))
    criterio_vistas            = bool(row.get("criterio_vistas", False))
    criterio_buen_postre       = bool(row.get("criterio_buen_postre", False))
    criterio_precio_calidad    = bool(row.get("criterio_precio_calidad", False))
    criterio_grupos_grandes    = bool(row.get("criterio_grupos_grandes", False))
    criterio_vegano_veg        = bool(row.get("criterio_vegano_vegetariano", False))
    criterio_sin_gluten        = bool(row.get("criterio_sin_gluten", False))

    # Derivar criterios del Proyecto A desde dimensiones NLP del Proyecto B
    # buena_comida: % positivo en dimensión comida
    comida_pos = float(row.get("comida_pos", 0) or 0)
    comida_menciones = float(row.get("comida_menciones", 1) or 1)
    buena_comida = (comida_pos / max(comida_menciones, 1)) > 0.7 if comida_menciones >= 3 else float(row.get("pct_positivo", 0) or 0) >= 80

    # buen_servicio
    serv_pos = float(row.get("servicio_pos", 0) or 0)
    serv_menciones = float(row.get("servicio_menciones", 1) or 1)
    buen_servicio = (serv_pos / max(serv_menciones, 1)) > 0.7 if serv_menciones >= 3 else False

    # buen_ambiente
    amb_avg = float(row.get("ambiente_avg_stars", 0) or 0)
    buen_ambiente = amb_avg >= 4.0 if amb_avg > 0 else False

    # espera_corta
    vel_neg = float(row.get("velocidad_neg", 0) or 0)
    espera_corta = vel_neg == 0

    # buena_relacion_precio_calidad
    precio_avg = float(row.get("precio_avg_stars", 0) or 0)
    buena_relacion_pq = precio_avg >= 4.0 if precio_avg > 0 else False

    # Rango de precio: intentar inferir de precio_avg_stars
    rango_precio = ""
    if precio_avg > 0:
        if precio_avg >= 4.5:
            rango_precio = "euro"
        elif precio_avg >= 4.0:
            rango_precio = "euro euro"
        elif precio_avg >= 3.0:
            rango_precio = "euro euro euro"
        else:
            rango_precio = "euro euro euro euro"

    r = {
        "nombre":                       str(row.get("nombre_display", row.get("nombre", "")) or ""),
        "valoracion":                   float(row.get("valoracion_display", 0) or 0),
        "votaciones":                   int(row.get("votaciones", 0) or 0),
        "direccion":                    str(row.get("direccion_display", row.get("direccion", "")) or ""),
        "resumen":                      resumen,
        "rango_precio":                 rango_precio,
        "dato_curioso":                 "",  # no existe en Proyecto B
        "aspectos_positivos":           [],  # no existe en Proyecto B — el React lo muestra si está
        "aspectos_negativos":           [],
        "platos_destacados":            platos_destacados,
        "platos_frecuencia":            json.dumps(platos_frecuencia, ensure_ascii=False),
        "perfil_cliente":               "{}",
        # Booleanos que espera el modal del React
        "buena_comida":                 buena_comida,
        "buen_servicio":                buen_servicio,
        "buen_ambiente":                buen_ambiente,
        "espera_corta":                 espera_corta,
        "buena_relacion_precio_calidad": buena_relacion_pq,
        "apto_ninos":                   criterio_ninos,
        "apto_mascotas":                criterio_mascotas,
        "terraza_exterior":             criterio_terraza,
        "recomendable_en_pareja":       criterio_romantico,
        "buenas_vistas":                criterio_vistas,
        "acceso_minusvalidos":          False,  # no existe en Proyecto B
        # Criterios nuevos — positivos
        "buen_postre":                  criterio_buen_postre,
        "buena_relacion_calidad_precio": criterio_precio_calidad,
        "apto_grupos":                  criterio_grupos_grandes,
        "opciones_veganas":             criterio_vegano_veg,
        "apto_celiaco":                 criterio_sin_gluten,
        # Advertencias — negativos (se muestran en rojo en Ver detalles)
        "aviso_espera_larga":           float(row.get("velocidad_neg", 0) or 0) > 4,
        "aviso_precio_elevado":         float(row.get("precio_neg", 0) or 0) > 4,
        "aviso_servicio_mejorable":     float(row.get("servicio_neg", 0) or 0) > 6,
        # Frases de reseñas que justifican cada criterio
        "frases_criterios": {
            k: str(row.get(f"criterio_{k}_frases", "") or "")
            for k in ["ninos", "mascotas", "terraza", "vistas", "musica_directo",
                      "romantico", "buen_postre", "precio_calidad",
                      "grupos_grandes", "vegano_vegetariano", "sin_gluten"]
            if str(row.get(f"criterio_{k}_frases", "") or "").strip()
            and str(row.get(f"criterio_{k}_frases", "")).strip().lower() not in ("nan", "none", "")
        },
        "servicio_frases": "" if str(row.get("servicio_frases", "") or "").strip().lower() in ("nan","none","") else str(row.get("servicio_frases", "") or ""),
        "personal_destacado": "" if str(row.get("personal_destacado", "") or "").strip().lower() in ("nan","none","") else str(row.get("personal_destacado", "") or ""),
        "resenas_destacadas": "" if str(row.get("resenas_destacadas", "") or "").strip().lower() in ("nan","none","") else str(row.get("resenas_destacadas", "") or ""),
        # Score NLP para ordenación interna
        "_pct_positivo":                float(row.get("pct_positivo", 0) or 0),
        "_avg_estrellas":               float(row.get("avg_estrellas_modelo", 0) or 0),
    }

    # Coordenadas
    lat = row.get("latitud")
    lon = row.get("longitud")
    if pd.notna(lat) and pd.notna(lon):
        r["latitud"]  = float(lat)
        r["longitud"] = float(lon)
    if distancia_km is not None:
        r["distancia_km"] = round(distancia_km, 2)

    return r


# ═══════════════════════════════════════════════════════════════════════════════
# BÚSQUEDA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS_BUSQUEDA = {
    "quiero", "busco", "algo", "cerca", "de", "en", "por", "para", "con",
    "un", "una", "unos", "unas", "los", "las", "el", "la", "que", "hay",
    "donde", "puedo", "comer", "tomar", "ver", "ir", "dame", "dime",
    "mucho", "muy", "bien", "buen", "buena", "buenos", "buenas",
    "necesito", "pon", "recomienda",
    "restaurante", "restaurantes", "sitio", "sitios", "lugar", "lugares",
}


def _buscar(consulta: str) -> tuple[list, dict]:
    """
    Motor de búsqueda principal. Devuelve (restaurantes_ordenados, meta).
    Replica la lógica JS de generar_agente.py en Python.
    """
    if df_global is None:
        return [], {}

    df = df_global.copy()
    consulta_norm = _norm(consulta)
    tokens = [t for t in re.split(r'[\s,.()/\-]+', consulta_norm) if len(t) >= 3]

    # Detectar intención
    cocina = _detectar_cocina(consulta)
    zona_coords = _detectar_zona(consulta)
    criterios = _detectar_criterios(consulta)

    # Tokens de zona — se excluyen del score de plato para evitar falsos positivos
    # Ej: "croquetas cerca de malasaña" → tokens_zona=["malasana"], tokens_plato=["croquetas"]
    tokens_zona = set()
    if zona_coords:
        for zona_key in ZONAS_MADRID:
            if _norm(zona_key) in _norm(consulta):
                tokens_zona.update(_norm(zona_key).split())
    tokens_plato_candidatos = [
        t for t in tokens
        if t not in tokens_zona and t not in STOPWORDS_BUSQUEDA and len(t) >= 4
    ]
    # Validar que cada token candidato es un plato real en el CSV
    # Si hay cocina detectada, aceptar también tokens que sean platos propios de esa cocina
    # (evita que "chuleton" falle cuando "asador" ya detectó cocina española)
    def _es_plato_valido(t, cocina_detectada):
        if df.apply(lambda r: _score_texto(r, [t]), axis=1).max() >= 2.5:
            return True
        if cocina_detectada:
            platos_cocina = [_norm(p) for p in COCINAS.get(cocina_detectada, [])]
            if any(t in p or p in t for p in platos_cocina if len(p) >= 4):
                return True
        return False

    tokens_plato = [
        t for t in tokens_plato_candidatos
        if _es_plato_valido(t, cocina)
    ]

    # Calcular score base de calidad para todos
    df["_score_calidad"] = df.apply(_score_calidad, axis=1)

    # ── Score por texto/platos (solo tokens de plato, sin tokens de zona) ──────
    if cocina:
        df["_score_match"] = df.apply(lambda r: _score_cocina(r, cocina), axis=1)
    elif tokens_plato:
        df["_score_match"] = df.apply(lambda r: _score_texto(r, tokens_plato), axis=1)
    else:
        df["_score_match"] = df.apply(lambda r: _score_texto(r, tokens), axis=1)

    # Buscar también por nombre del restaurante
    df["_score_nombre"] = df["nombre_display"].apply(
        lambda n: 3.0 if any(t in _norm(str(n)) for t in tokens_plato if len(t) >= 3) else 0.0
    )

    # ── Score de distancia (si hay zona) ──────────────────────────
    if zona_coords:
        def _dist(row):
            lat, lon = row.get("latitud"), row.get("longitud")
            if pd.notna(lat) and pd.notna(lon):
                return _haversine(zona_coords, (float(lat), float(lon)))
            return 999.0
        df["_dist_zona"] = df.apply(_dist, axis=1)
        max_dist = df["_dist_zona"].replace(999.0, pd.NA).dropna().max() or 10.0
        df["_score_dist"] = (1 - (df["_dist_zona"] / max_dist).clip(0, 1)) * 10
    else:
        df["_dist_zona"] = None
        df["_score_dist"] = 0.0

    # ── Score final combinado ──────────────────────────────────────
    # hay_plato: solo True si hay tokens de plato con score real en el CSV
    hay_plato = bool(tokens_plato) and df["_score_match"].max() >= 2.5

    # Si el usuario pidió un plato/cocina concreto pero ningún restaurante lo tiene
    # → devolver vacío con mensaje claro, nunca un ranking por calidad general
    if tokens_plato and not hay_plato and not cocina:
        return [], {
            "cocina": None, "zona": zona_coords, "criterios": criterios,
            "n_total": 0, "tokens_plato": tokens_plato, "hay_plato": False,
            "sin_resultados_plato": True,
        }

    if zona_coords and hay_plato:
        # Zona + plato: el plato filtra, la distancia ordena
        # 70% distancia + 30% calidad (el plato ya filtra hard abajo)
        df["_score_final"] = (
            df["_score_dist"]    * 0.70 +
            df["_score_calidad"] * 0.30
        )
    elif zona_coords:
        # Solo zona: distancia manda
        # 70% distancia + 30% calidad
        df["_score_final"] = (
            df["_score_dist"]    * 0.70 +
            df["_score_calidad"] * 0.30
        )
    elif cocina or df["_score_match"].max() > 4:
        # Con match de texto/cocina: 50% match + 50% calidad
        df["_score_final"] = (
            (df["_score_match"] + df["_score_nombre"]).clip(0, 10) * 0.50 +
            df["_score_calidad"] * 0.50
        )
    else:
        # Sin ningún anclaje — no devolver nada
        df["_score_final"] = df["_score_calidad"]

    # ── Filtrar por criterios detectados ──────────────────────────
    df_filtrado = df.copy()
    for criterio in criterios:
        mascara = df_filtrado.apply(lambda r: _pasa_criterio(r, criterio), axis=1)
        if mascara.sum() > 0:
            df_filtrado = df_filtrado[mascara]

    # ── Si con filtros quedan pocos, usar sin filtros ──────────────
    if len(df_filtrado) < 3 and len(criterios) > 0:
        df_filtrado = df.copy()

    # ── Filtrar hard por cocina — sin excepciones ────────────────────────────
    if cocina:
        df_con_match = df_filtrado[df_filtrado["_score_match"] > 0]
        df_filtrado = df_con_match

    # ── Filtrar hard por plato (siempre que haya plato detectado) ──────────────
    # Umbral >= 2.5 para asegurar que el plato aparece como tal en reseñas
    # Aplica tanto con zona como sin ella: "asador con chuletón" → solo asadores con chuletón
    UMBRAL_PLATO = 2.5
    if hay_plato:
        # Cuando hay cocina Y plato: recalcular score_match solo por plato (no por cocina)
        # para que el filtro sea sobre el plato concreto, no sobre la cocina
        if cocina:
            df_filtrado["_score_plato"] = df_filtrado.apply(
                lambda r: _score_texto(r, tokens_plato), axis=1
            )
            df_con_plato = df_filtrado[df_filtrado["_score_plato"] >= UMBRAL_PLATO]
        else:
            df_con_plato = df_filtrado[df_filtrado["_score_match"] >= UMBRAL_PLATO]
        if len(df_con_plato) >= 2:
            df_filtrado = df_con_plato
        elif len(df_con_plato) == 1:
            df_filtrado = df_con_plato  # aunque sea 1, es correcto
        # Si 0 resultados con el plato → devolver vacío con mensaje claro
        elif len(df_con_plato) == 0:
            return [], {
                "cocina": cocina, "zona": zona_coords, "criterios": criterios,
                "n_total": 0, "tokens_plato": tokens_plato, "hay_plato": True,
                "sin_resultados_plato": True,
            }

    # ── Ordenar y tomar top N ──────────────────────────────────────
    n_resultados = 6 if (cocina or tokens or criterios) else 8
    if zona_coords and "_dist_zona" in df_filtrado.columns:
        # Con zona: ordenar por distancia física real, no por score
        df_top = (
            df_filtrado[df_filtrado["_dist_zona"] < 999.0]
            .sort_values("_dist_zona", ascending=True)
            .head(n_resultados)
        )
        if len(df_top) < 2:
            # Fallback si no hay suficientes con coords
            df_top = df_filtrado.sort_values("_score_final", ascending=False).head(n_resultados)
    else:
        df_top = df_filtrado.sort_values("_score_final", ascending=False).head(n_resultados)

    restaurantes = []
    for _, row in df_top.iterrows():
        dist_km = None
        if zona_coords and pd.notna(row.get("_dist_zona")) and row["_dist_zona"] < 999.0:
            dist_km = row["_dist_zona"]
        restaurantes.append(_fila_a_restaurante(row, distancia_km=dist_km))

    # ── Sin ningún anclaje reconocible → pedir que reformule ─────────────────
    # No devolver ranking por calidad si el usuario no especificó nada concreto
    sin_anclaje = (
        not cocina
        and not zona_coords
        and not criterios
        and not hay_plato
    )
    if sin_anclaje:
        return [], {
            "cocina": None, "zona": None, "criterios": [],
            "n_total": 0, "tokens_plato": [], "hay_plato": False,
            "sin_anclaje": True,
        }

    meta = {
        "cocina":       cocina,
        "zona":         zona_coords,
        "criterios":    criterios,
        "n_total":      len(df_filtrado),
        "tokens_plato": tokens_plato,
        "hay_plato":    hay_plato,
    }
    return restaurantes, meta


# ═══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE RESPUESTA EN LENGUAJE NATURAL
# ═══════════════════════════════════════════════════════════════════════════════

def _generar_respuesta(consulta: str, restaurantes: list, meta: dict) -> str:
    """
    Genera la respuesta en texto/markdown a partir de los restaurantes encontrados.
    Formato idéntico al que espera el componente MensajeCompacto del React.
    No usa LLM — es determinista y rápido.
    """
    if not restaurantes:
        cocina_vacia = meta.get("cocina")
        sin_resultados_plato = meta.get("sin_resultados_plato", False)
        tokens_plato = meta.get("tokens_plato", [])

        if cocina_vacia:
            return (
                f"No tengo restaurantes de cocina {cocina_vacia} con suficientes platos "
                f"identificativos en mi base de datos. "
                f"Prueba con 'restaurante gallego', 'pulpo a feira' o 'empanada gallega'."
            )
        if sin_resultados_plato and tokens_plato:
            plato = tokens_plato[0]
            zona_txt = " cerca de esa zona" if meta.get("zona") else ""
            return (
                f"No he encontrado restaurantes que destaquen por {plato}{zona_txt}. "
                f"Prueba con un término diferente o amplía la zona de búsqueda."
            )
        if meta.get("sin_anclaje"):
            return (
                "No he entendido bien tu búsqueda. "
                "Puedes preguntarme por una cocina (italiana, gallega, japonesa...), "
                "un plato (croquetas, pulpo, sushi...), una zona de Madrid (Malasaña, Retiro, Chamberí...) "
                "o una situación (romántico, con niños, con terraza...). ¿Qué estás buscando?"
            )
        return (
            "No he encontrado restaurantes que encajen con tu búsqueda. "
            "Prueba con otro plato, cocina o zona de Madrid."
        )

    cocina    = meta.get("cocina")
    zona      = meta.get("zona")
    criterios = meta.get("criterios", [])

    # Intro contextual
    intro_parts = []
    if cocina:
        intro_parts.append(f"cocina {cocina}")
    if criterios:
        etiquetas = {
            "romantico":          "ambiente romántico",
            "ninos":              "apto para niños",
            "mascotas":           "que admiten mascotas",
            "terraza":            "con terraza",
            "vistas":             "con vistas",
            "musica_directo":     "con música en directo",
            "tranquilo":          "tranquilos",
            "precio_ok":          "con buen precio",
            "muy_valorado":       "muy valorados",
            "buen_postre":        "con buenos postres",
            "precio_calidad":     "con buena relación calidad-precio",
            "grupos_grandes":     "aptos para grupos o celebraciones",
            "vegano_vegetariano": "con opciones veganas o vegetarianas",
            "sin_gluten":         "con opciones sin gluten",
        }
        intro_parts.append(", ".join(etiquetas.get(c, c) for c in criterios))

    tokens_plato = meta.get("tokens_plato", [])
    hay_plato    = meta.get("hay_plato", False)

    lineas = [""]

    for r in restaurantes:
        nombre  = r["nombre"]
        val     = r["valoracion"]
        pct     = r.get("_pct_positivo", 0)
        dist    = r.get("distancia_km")
        platos  = r.get("platos_destacados", [])
        resumen = r.get("resumen", "")

        # Encabezado de tarjeta
        lineas.append(f"**{nombre}**")

        # Valoración + % positivo
        meta_linea = f"Valoración: {val}"
        if pct:
            meta_linea += f" · {pct:.0f}% reseñas positivas"
        if r.get("rango_precio"):
            from_mapa = {"euro": "€", "euro euro": "€€", "euro euro euro": "€€€", "euro euro euro euro": "€€€€"}
            precio_fmt = from_mapa.get(r["rango_precio"], r["rango_precio"])
            meta_linea += f" · {precio_fmt}"
        lineas.append(meta_linea)

        # Distancia si hay zona
        if dist is not None:
            lineas.append(f"📍 A {dist:.2f} km")

        # Resumen breve
        if resumen:
            # Truncar a 120 chars para que quede limpio
            resumen_corto = resumen[:120] + ("..." if len(resumen) > 120 else "")
            lineas.append(resumen_corto)

        # Platos destacados (con menciones si disponibles)
        if platos:
            platos_frec = {}
            try:
                platos_frec = json.loads(r.get("platos_frecuencia", "{}") or "{}")
            except Exception:
                pass

            platos_fmt = []
            for p in platos[:4]:
                n_menciones = platos_frec.get(p)
                if n_menciones:
                    platos_fmt.append(f"{p} ({n_menciones}/90 reseñas)")
                else:
                    platos_fmt.append(p)
            lineas.append(f"🍽️ Platos: {', '.join(platos_fmt)}")

        # Badges de criterios
        badges = []
        if r.get("apto_ninos"):      badges.append("👶 Apto niños")
        if r.get("apto_mascotas"):   badges.append("🐾 Mascotas")
        if r.get("terraza_exterior"):badges.append("☀️ Terraza")
        if r.get("buenas_vistas"):   badges.append("🏙️ Con vistas")
        if r.get("recomendable_en_pareja"): badges.append("🕯️ Romántico")
        if badges:
            lineas.append("  ".join(badges))

        lineas.append("")  # Separador entre restaurantes

    return "\n".join(lineas).strip()



# ═══════════════════════════════════════════════════════════════════════════════
# RAG — INDEXACIÓN (ejecutar una vez: python main.py --indexar)
# ═══════════════════════════════════════════════════════════════════════════════

CHROMA_DIR       = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME  = "restaurantes_resenas"
MAX_RESENAS_RAG  = 40   # reseñas por restaurante incluidas en el índice
GEMINI_EMBED_MODEL = "models/embedding-001"  # gratuito, 1500 req/día, sin descarga

# Instancia global de la colección ChromaDB (carga lazy en primer uso)
_chroma_collection = None


def _gemini_embed(textos: list[str]) -> list[list[float]]:
    """
    Genera embeddings usando Gemini embedding-001.
    Gratuito (1500 req/día), sin descarga de modelo, funciona en Render free.
    Devuelve lista de vectores float. Si falla, devuelve vectores vacíos.
    """
    if not GEMINI_API_KEY:
        return [[] for _ in textos]

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{GEMINI_EMBED_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    )

    # Gemini embedding acepta lotes de hasta 100
    LOTE = 100
    todos_vectores = []

    for i in range(0, len(textos), LOTE):
        lote = textos[i:i + LOTE]
        requests_body = {
            "requests": [
                {
                    "model": GEMINI_EMBED_MODEL,
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                }
                for t in lote
            ]
        }
        data = json.dumps(requests_body).encode()
        try:
            req  = _ureq.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
            resp = _ureq.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            vectores = [e["values"] for e in result.get("embeddings", [])]
            todos_vectores.extend(vectores)
        except Exception as e:
            print(f"  ⚠️  Error embedding Gemini: {e}")
            todos_vectores.extend([[] for _ in lote])
        time.sleep(0.1)  # respetar rate limit

    return todos_vectores


def _gemini_embed_query(texto: str) -> list[float]:
    """Embedding para una consulta (taskType RETRIEVAL_QUERY)."""
    if not GEMINI_API_KEY:
        return []
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    )
    data = json.dumps({
        "model": GEMINI_EMBED_MODEL,
        "content": {"parts": [{"text": texto}]},
        "taskType": "RETRIEVAL_QUERY",
    }).encode()
    try:
        req  = _ureq.Request(url, data=data,
                             headers={"Content-Type": "application/json"}, method="POST")
        resp = _ureq.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        return result.get("embedding", {}).get("values", [])
    except Exception:
        return []


def _rag_construir_documento(row: pd.Series, resenas: list) -> str:
    """
    Texto que se indexa en ChromaDB para cada restaurante.
    Combina el resumen estructurado del CSV con fragmentos de reseñas.
    Compacto a propósito: chunks más pequeños = mejor recall semántico.
    """
    lineas = [f"Restaurante: {row.get('nombre', '')}"]

    if row.get("cocina_detectada"):
        lineas.append(f"Cocina: {row['cocina_detectada']}")
    if row.get("tipo_establecimiento"):
        lineas.append(f"Tipo: {row['tipo_establecimiento']}")
    if row.get("top5_platos"):
        lineas.append(f"Platos destacados: {row['top5_platos']}")
    if row.get("personal_destacado"):
        lineas.append(f"Personal destacado: {row['personal_destacado']}")

    criterios = [
        c.replace("criterio_", "")
        for c in ["criterio_ninos", "criterio_mascotas", "criterio_terraza",
                  "criterio_vistas", "criterio_musica_directo", "criterio_romantico"]
        if str(row.get(c, "")).lower() == "true"
    ]
    if criterios:
        lineas.append(f"Características: {', '.join(criterios)}")

    if row.get("avg_estrellas_modelo"):
        lineas.append(
            f"Valoración modelo: {row['avg_estrellas_modelo']} estrellas "
            f"({row.get('pct_positivo', 0)}% positivo)"
        )

    if resenas:
        lineas.append("\nOpiniones:")
        for r in resenas[:MAX_RESENAS_RAG]:
            texto_limpio = re.sub(r'\s+', ' ', str(r)).strip()[:280]
            lineas.append(f"- {texto_limpio}")

    return "\n".join(lineas)


def rag_indexar(verbose: bool = True):
    """
    Lee analisis_restaurantes.csv + resenas_unificadas.csv,
    genera embeddings locales y persiste en ChromaDB.
    Solo necesita ejecutarse una vez (o al añadir nuevas reseñas).
    """
    try:
        import chromadb
    except ImportError:
        print("ERROR: instala chromadb:  pip install chromadb langgraph langchain-core")
        return

    csv_analisis = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
    csv_resenas  = os.path.join(BASE_DIR, "resenas_unificadas.csv")

    if not os.path.exists(csv_analisis):
        print(f"ERROR: No se encuentra {csv_analisis}")
        return

    if verbose:
        print("Cargando CSVs...")
    df_an = pd.read_csv(csv_analisis)
    df_an["id_restaurante"] = df_an["id_restaurante"].astype(str)

    df_res = pd.DataFrame()
    col_id_res = col_texto = None
    if os.path.exists(csv_resenas):
        df_res = pd.read_csv(csv_resenas)
        for c in ["Id_Restaurante", "id_restaurante", "id"]:
            if c in df_res.columns:
                col_id_res = c
                break
        for c in ["Review", "review", "texto", "Texto"]:
            if c in df_res.columns:
                col_texto = c
                break
        if col_id_res:
            df_res[col_id_res] = df_res[col_id_res].astype(str)

    if verbose:
        print(f"  {len(df_an)} restaurantes | {len(df_res)} reseñas")

    if not GEMINI_API_KEY:
        print("ERROR: necesitas GEMINI_API_KEY para indexar con embeddings de Gemini.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        if verbose:
            print("  Colección anterior eliminada. Reindexando...")
    except Exception:
        pass

    # ChromaDB sin embedding_function — gestionamos los embeddings manualmente
    col = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs, ids, metas = [], [], []
    for _, row in df_an.iterrows():
        rid = str(row["id_restaurante"])
        resenas = []
        if col_id_res and col_texto and not df_res.empty:
            resenas = (
                df_res[df_res[col_id_res] == rid][col_texto]
                .dropna().astype(str).tolist()
            )
        doc = _rag_construir_documento(row, resenas)
        docs.append(doc)
        ids.append(f"rest_{rid}")
        metas.append({
            "id_restaurante": rid,
            "nombre":         str(row.get("nombre", "")),
            "cocina":         str(row.get("cocina_detectada", "") or ""),
            "valoracion":     float(row.get("avg_estrellas_modelo", 0) or 0),
            "pct_positivo":   float(row.get("pct_positivo", 0) or 0),
        })

    # Generar embeddings con Gemini en lotes y añadir a ChromaDB
    LOTE = 50
    total = len(docs)
    if verbose:
        print(f"Generando embeddings y indexando {total} restaurantes (lotes de {LOTE})...")
    for i in range(0, total, LOTE):
        fin = min(i + LOTE, total)
        lote_docs  = docs[i:fin]
        lote_ids   = ids[i:fin]
        lote_metas = metas[i:fin]
        embeddings = _gemini_embed(lote_docs)
        # Filtrar entradas sin embedding válido
        validos = [(d, id_, m, e) for d, id_, m, e in
                   zip(lote_docs, lote_ids, lote_metas, embeddings) if e]
        if validos:
            col.add(
                documents  = [v[0] for v in validos],
                ids        = [v[1] for v in validos],
                metadatas  = [v[2] for v in validos],
                embeddings = [v[3] for v in validos],
            )
        if verbose:
            print(f"  [{fin}/{total}]")

    if verbose:
        print(f"✔ Indexación completa → {CHROMA_DIR}")



# ═══════════════════════════════════════════════════════════════════════════════
# RAG — RETRIEVAL + GEMINI FLASH
# Gemini solo se llama cuando el NLP no encuentra resultados relevantes.
# Caché en disco: consultas repetidas = coste cero.
# Coste estimado por consulta RAG: ~0.009 € (900 tokens input, 400 output)
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash"
CACHE_RAG_PATH  = os.path.join(BASE_DIR, ".cache_rag_respuestas.json")
N_DOCS_RAG      = 5       # documentos a recuperar de ChromaDB
MAX_TOKENS_RAG  = 400     # tokens máximos de output Gemini

# Caché en memoria — se carga una sola vez al arrancar
_cache_rag: dict = {}


def _rag_cargar_cache():
    global _cache_rag
    if os.path.exists(CACHE_RAG_PATH):
        try:
            with open(CACHE_RAG_PATH, "r", encoding="utf-8") as f:
                _cache_rag = json.load(f)
        except Exception:
            _cache_rag = {}


def _rag_guardar_cache():
    try:
        with open(CACHE_RAG_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache_rag, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _rag_clave(consulta: str) -> str:
    """Normaliza la consulta para usarla como clave de caché."""
    c = consulta.lower().strip()
    c = unicodedata.normalize("NFD", c)
    c = "".join(ch for ch in c if unicodedata.category(ch) != "Mn")
    return re.sub(r"[\W_]+", " ", c).strip()


def _rag_get_collection():
    """Carga ChromaDB de forma lazy (solo en el primer uso). Sin modelo local."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        import chromadb
    except ImportError:
        return None

    if not os.path.exists(CHROMA_DIR):
        return None

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        # Sin embedding_function: los embeddings se pasan manualmente en cada query
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        _chroma_collection = None
    return _chroma_collection


def _rag_buscar_semantico(consulta: str) -> list:
    """Devuelve los N documentos más similares usando embedding de Gemini."""
    col = _rag_get_collection()
    if col is None:
        return []
    embedding = _gemini_embed_query(consulta)
    if not embedding:
        return []
    try:
        results = col.query(query_embeddings=[embedding], n_results=N_DOCS_RAG)
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id":        results["ids"][0][i],
                "documento": results["documents"][0][i],
                "metadata":  results["metadatas"][0][i],
                "distancia": results["distances"][0][i],
            })
        return docs
    except Exception:
        return []


def _rag_gemini_call(prompt: str) -> str:
    """Llamada directa a Gemini Flash. Sin dependencias externas (solo urllib)."""
    if not GEMINI_API_KEY:
        return ""
    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": MAX_TOKENS_RAG},
    }).encode()
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    for intento in range(3):
        try:
            req  = _ureq.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
            resp = _ureq.urlopen(req, timeout=25)
            result = json.loads(resp.read())
            partes = result["candidates"][0]["content"]["parts"]
            return " ".join(p.get("text", "") for p in partes if p.get("text")).strip()
        except Exception as e:
            if any(c in str(e) for c in ["503", "429", "500"]) and intento < 2:
                time.sleep(2 ** (intento + 1))
            else:
                return ""
    return ""


def _rag_construir_prompt(consulta: str, docs: list, historial: list) -> str:
    """Prompt compacto para minimizar tokens de entrada."""
    contexto = "\n\n---\n\n".join(d["documento"][:650] for d in docs)

    hist_texto = ""
    if historial:
        turnos = historial[-3:]  # máx 3 turnos para no inflar el prompt
        hist_texto = "\nHistorial:\n" + "\n".join(
            f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:180]}"
            for m in turnos
        ) + "\n"

    return (
        "Eres un asistente de recomendación de restaurantes de Madrid. "
        "Responde en español, de forma breve y conversacional (máximo 4-5 oraciones). "
        "Si la consulta no encaja con el contexto, dilo con naturalidad. "
        "No inventes datos.\n"
        f"{hist_texto}\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {consulta}\n\nRespuesta:"
    )


def rag_responder(consulta: str, historial: list) -> dict:
    """
    Punto de entrada del RAG. Busca en ChromaDB y genera con Gemini.
    Devuelve {"respuesta": str, "restaurantes": list, "desde_cache": bool}.
    """
    clave = _rag_clave(consulta)

    # ── Caché hit: coste cero ──────────────────────────────────────────────────
    if clave in _cache_rag:
        entrada = _cache_rag[clave]
        return {
            "respuesta":    entrada["respuesta"],
            "restaurantes": entrada.get("restaurantes", []),
            "desde_cache":  True,
        }

    # ── Retrieval semántico ────────────────────────────────────────────────────
    docs = _rag_buscar_semantico(consulta)

    if not docs:
        return {
            "respuesta": (
                "No encontré restaurantes que encajen con tu búsqueda. "
                "Prueba con un tipo de cocina o plato concreto."
            ),
            "restaurantes": [],
            "desde_cache":  False,
        }

    # ── Generación con Gemini (o fallback sin API key) ─────────────────────────
    if GEMINI_API_KEY:
        prompt    = _rag_construir_prompt(consulta, docs, historial)
        respuesta = _rag_gemini_call(prompt)
    else:
        respuesta = ""

    if not respuesta:
        nombres   = [d["metadata"].get("nombre", "—") for d in docs[:3]]
        respuesta = (
            "He encontrado estos restaurantes que pueden interesarte: "
            + ", ".join(nombres) + ". "
            "Para más detalle, prueba a ser más específico en tu búsqueda."
        )

    restaurantes_meta = [
        {
            "id_restaurante": d["metadata"].get("id_restaurante", ""),
            "nombre":         d["metadata"].get("nombre", ""),
            "cocina":         d["metadata"].get("cocina", ""),
            "valoracion":     d["metadata"].get("valoracion", 0),
            "similitud":      round(1 - d["distancia"], 3),
        }
        for d in docs
    ]

    # ── Guardar en caché ───────────────────────────────────────────────────────
    _cache_rag[clave] = {"respuesta": respuesta, "restaurantes": restaurantes_meta}
    _rag_guardar_cache()

    return {"respuesta": respuesta, "restaurantes": restaurantes_meta, "desde_cache": False}



# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE LANGGRAPH
# Grafo: interpretar → nlp_search → router → [sintetizar | rag_search → sintetizar]
#
# Router activa RAG cuando:
#   · El NLP devuelve < 2 resultados con score relevante
#   · La consulta no tiene ningún anclaje (cocina / zona / criterio)
# ═══════════════════════════════════════════════════════════════════════════════

# Umbrales del router — ajusta según tu distribución de scores
UMBRAL_SCORE_NLP  = 1.0   # score mínimo para considerar NLP suficiente
MIN_RESULTS_NLP   = 2     # resultados mínimos para evitar RAG


class EstadoAgente(TypedDict):
    consulta:          str
    historial:         list          # [{"role": "user"/"assistant", "content": str}]
    restaurantes_nlp:  Optional[list]
    meta_nlp:          Optional[dict]
    score_nlp:         float
    usar_rag:          bool
    respuesta:         str
    restaurantes:      list
    proyecto:          str           # "nlp" | "rag" | "rag_cache"


def _agente_nodo_interpretar(estado: EstadoAgente) -> dict:
    """Inicializa el estado. Punto de extensión para preprocesado futuro."""
    return {
        "restaurantes_nlp": None,
        "meta_nlp":         None,
        "score_nlp":        0.0,
        "usar_rag":         False,
        "respuesta":        "",
        "restaurantes":     [],
        "proyecto":         "nlp",
    }


def _agente_nodo_nlp_search(estado: EstadoAgente) -> dict:
    """Motor NLP determinista — gratuito, sin ninguna API."""
    try:
        restaurantes, meta = _buscar(estado["consulta"])
        score_max = max((r.get("_score_final", 0) for r in restaurantes), default=0.0)
    except Exception:
        restaurantes, meta, score_max = [], {}, 0.0
    return {"restaurantes_nlp": restaurantes, "meta_nlp": meta, "score_nlp": score_max}


def _agente_nodo_router(estado: EstadoAgente) -> dict:
    """Decide si los resultados NLP son suficientes o hay que tirar de RAG."""
    nlp = estado.get("restaurantes_nlp") or []
    meta  = estado.get("meta_nlp") or {}
    score = estado.get("score_nlp", 0.0)

    sin_ancla = (
        not meta.get("cocina")
        and not meta.get("zona")
        and not meta.get("criterios")
    )

    usar_rag = (
        len(nlp) < MIN_RESULTS_NLP
        or score < UMBRAL_SCORE_NLP
        or sin_ancla
    )
    return {"usar_rag": usar_rag}


def _agente_nodo_rag(estado: EstadoAgente) -> dict:
    """Búsqueda semántica + Gemini (solo cuando el router lo decide)."""
    resultado = rag_responder(estado["consulta"], estado["historial"])
    proyecto  = "rag_cache" if resultado.get("desde_cache") else "rag"
    return {
        "respuesta":    resultado["respuesta"],
        "restaurantes": resultado["restaurantes"],
        "proyecto":     proyecto,
    }


def _agente_nodo_sintetizar(estado: EstadoAgente) -> dict:
    """Formatea la respuesta final. Si venimos del NLP, genera el texto (gratuito)."""
    if not estado.get("usar_rag"):
        restaurantes = estado.get("restaurantes_nlp") or []
        meta         = estado.get("meta_nlp") or {}
        # Limpiar campos internos antes de devolver al frontend
        restaurantes_limpios = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in restaurantes
        ]
        return {
            "respuesta":    _generar_respuesta(estado["consulta"], restaurantes, meta),
            "restaurantes": restaurantes_limpios,
            "proyecto":     "nlp",
        }
    # RAG: la respuesta ya viene del nodo anterior
    return {}


def _agente_decide_camino(estado: EstadoAgente) -> str:
    return "rag" if estado.get("usar_rag") else "sintetizar"


def _construir_grafo_agente():
    """Compila el grafo LangGraph una sola vez al arrancar."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None

    g = StateGraph(EstadoAgente)
    g.add_node("interpretar", _agente_nodo_interpretar)
    g.add_node("nlp_search",  _agente_nodo_nlp_search)
    g.add_node("router",      _agente_nodo_router)
    g.add_node("rag_search",  _agente_nodo_rag)
    g.add_node("sintetizar",  _agente_nodo_sintetizar)

    g.set_entry_point("interpretar")
    g.add_edge("interpretar", "nlp_search")
    g.add_edge("nlp_search",  "router")
    g.add_conditional_edges(
        "router", _agente_decide_camino,
        {"rag": "rag_search", "sintetizar": "sintetizar"},
    )
    g.add_edge("rag_search", "sintetizar")
    g.add_edge("sintetizar", END)

    return g.compile()


# Instancia global del agente (None si LangGraph no está instalado)
_agente_grafo = None


def agente_consultar(consulta: str, historial: list | None = None) -> dict:
    """
    Punto de entrada principal del agente conversacional.
    Si LangGraph no está disponible, cae directamente al NLP (sin errores).
    """
    global _agente_grafo

    # Inicialización lazy del grafo
    if _agente_grafo is None:
        _agente_grafo = _construir_grafo_agente()

    historial = historial or []

    # ── Fallback sin LangGraph ────────────────────────────────────────────────
    if _agente_grafo is None:
        try:
            restaurantes, meta = _buscar(consulta)
            respuesta = _generar_respuesta(consulta, restaurantes, meta)
            restaurantes_limpios = [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in restaurantes
            ]
        except Exception:
            respuesta = "No pude procesar tu consulta. Prueba con otros términos."
            restaurantes_limpios = []
        return {"respuesta": respuesta, "restaurantes": restaurantes_limpios, "proyecto": "nlp"}

    # ── Agente LangGraph ──────────────────────────────────────────────────────
    estado_inicial: EstadoAgente = {
        "consulta":          consulta,
        "historial":         historial,
        "restaurantes_nlp":  None,
        "meta_nlp":          None,
        "score_nlp":         0.0,
        "usar_rag":          False,
        "respuesta":         "",
        "restaurantes":      [],
        "proyecto":          "nlp",
    }

    resultado = _agente_grafo.invoke(estado_inicial)
    return {
        "respuesta":    resultado.get("respuesta", ""),
        "restaurantes": resultado.get("restaurantes", []),
        "proyecto":     resultado.get("proyecto", "nlp"),
    }



# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    _inicializar()
    _rag_cargar_cache()
    yield

app.router.lifespan_context = lifespan


@app.get("/")
def root():
    rag_disponible = os.path.exists(CHROMA_DIR)
    return {
        "status":  "ok",
        "mensaje": "API Restaurantes Madrid — NLP local + RAG + LangGraph",
        "rag":     rag_disponible,
    }


@app.get("/health")
def health():
    return {
        "status":               "ok",
        "restaurantes_cargados": len(df_global) if df_global is not None else 0,
        "backend":              "nlptown + RAG + LangGraph",
        "rag_disponible":       os.path.exists(CHROMA_DIR),
        "cache_rag_entradas":   len(_cache_rag),
    }


@app.post("/recomendar", response_model=RecomendacionResponse)
def endpoint_recomendar(request: ConsultaRequest):
    if not request.consulta.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
    if df_global is None:
        raise HTTPException(status_code=503, detail="Datos no cargados aún, espera un momento")

    historial = [
        {"role": m.role, "content": m.content}
        for m in (request.historial or [])
    ]

    try:
        resultado = agente_consultar(
            consulta  = request.consulta,
            historial = historial,
        )
        return RecomendacionResponse(
            respuesta        = resultado["respuesta"],
            proyecto         = resultado["proyecto"],
            restaurantes     = resultado["restaurantes"],
            consulta_usuario = request.consulta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/restaurantes")
def listar_restaurantes():
    """Devuelve todos los restaurantes con coordenadas (para el mapa inicial)."""
    if df_global is None:
        raise HTTPException(status_code=503, detail="Datos no cargados")
    try:
        df_con_coords = df_global[df_global["latitud"].notna() & df_global["longitud"].notna()]
        result = []
        for _, row in df_con_coords.iterrows():
            result.append({
                "id_restaurante": str(row.get("id_restaurante", "")),
                "nombre":         str(row.get("nombre_display", row.get("nombre", "")) or ""),
                "valoracion":     float(row.get("valoracion_display", 0) or 0),
                "votaciones":     int(row.get("votaciones", 0) or 0),
                "direccion":      str(row.get("direccion_display", row.get("direccion", "")) or ""),
                "latitud":        float(row["latitud"]),
                "longitud":       float(row["longitud"]),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indexar")
def endpoint_indexar():
    """
    Reconstruye la base vectorial ChromaDB.
    Llamar solo cuando haya nuevas reseñas en resenas_unificadas.csv.
    """
    try:
        rag_indexar(verbose=False)
        global _chroma_collection
        _chroma_collection = None   # forzar recarga en el próximo uso
        return {"status": "ok", "mensaje": "Base vectorial reconstruida correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — python main.py --indexar
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--indexar" in sys.argv:
        print("Iniciando indexación RAG...")
        # Cargar .env si existe
        _env = os.path.join(BASE_DIR, ".env")
        if os.path.exists(_env):
            with open(_env) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        rag_indexar(verbose=True)
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

