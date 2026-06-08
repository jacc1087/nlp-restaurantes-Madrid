"""
main.py — Backend FastAPI para el sistema de recomendación de restaurantes de Madrid.
Lógica de búsqueda del Proyecto B (nlptown/bert + análisis NLP local).
Interfaz compatible con el frontend React del Proyecto A.

Modos de uso:
  uvicorn main:app --reload        → servidor de desarrollo
  uvicorn main:app --host 0.0.0.0  → producción (Railway)

Archivos necesarios en el mismo directorio:
  analisis_restaurantes.csv        → generado por analizar_todos_restaurantes.py
  ranking.csv                      → ranking original con Valoracion, Votaciones, Dirección
  .cache_coords.json               → caché de coordenadas (generado por generar_agente.py)

Variables de entorno opcionales:
  GEMINI_API_KEY                   → no se usa en runtime, solo por compatibilidad
"""

import os
import re
import json
import math
import unicodedata
import ast
from typing import List, Optional

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
                "criterio_vistas", "criterio_musica_directo", "criterio_romantico"]:
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
    "retiro": (40.4153, -3.6844), "parque del retiro": (40.4153, -3.6844),
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


# Caché en memoria para geocodificación de calles (evita llamadas repetidas)
_geocode_cache: dict = {}

def _geocodificar_calle(texto: str) -> Optional[tuple]:
    """Geocodifica una calle o lugar de Madrid usando Nominatim (OpenStreetMap)."""
    if texto in _geocode_cache:
        return _geocode_cache[texto]
    try:
        import urllib.request, json as _json, urllib.parse
        query = urllib.parse.quote(f"{texto}, Madrid, España")
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1&countrycodes=es"
        req = urllib.request.Request(url, headers={"User-Agent": "RestaurantesMadridTFM/1.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = _json.loads(r.read())
        if data:
            coords = (float(data[0]["lat"]), float(data[0]["lon"]))
            _geocode_cache[texto] = coords
            print(f"  [geocode] '{texto}' → {coords}")
            return coords
    except Exception as e:
        print(f"  [geocode] error: {e}")
    _geocode_cache[texto] = None
    return None

def _detectar_zona(consulta: str) -> Optional[tuple]:
    """Detecta si la consulta menciona una zona de Madrid y devuelve sus coordenadas.
    Si no está en el diccionario de zonas, intenta geocodificar la calle con Nominatim."""
    c = _norm(consulta)
    # Patrones de extracción de zona/calle
    patrones = [
        r'cerca de(?:l| la| los| las)?\s+([\w\s]+?)(?:\s+y|\s+quiero|\s+busco|\s+que|,|\.|$)',
        r'estoy en\s+([\w\s]+?)(?:\s+y|\s+quiero|\s+busco|,|\.|$)',
        r'calle\s+([\w\s]{2,30}?)(?:\s+quiero|\s+busco|\s+que|,|\.|$)',
        r'por la calle\s+([\w\s]{2,30}?)(?:\s+quiero|\s+busco|,|\.|$)',
        r'por\s+([\w\s]{3,20}?)(?:\s+quiero|\s+busco|,|\.|$)',
        r'en\s+([\w\s]{3,30}?)(?:\s+quiero|\s+busco|\s+que|,|\.|$)',
        r'barrio de\s+([\w\s]+?)(?:,|\.|$)',
        r'zona de\s+([\w\s]+?)(?:,|\.|$)',
    ]
    for patron in patrones:
        m = re.search(patron, c)
        if m:
            zona_raw = m.group(1).strip()
            # 1. Buscar exacto en diccionario
            if zona_raw in ZONAS_MADRID:
                return ZONAS_MADRID[zona_raw]
            # 2. Búsqueda parcial en diccionario
            for nombre, coords in ZONAS_MADRID.items():
                if zona_raw in nombre or nombre in zona_raw:
                    return coords
            # 3. Geocodificar como calle de Madrid
            if len(zona_raw) >= 4:
                coords = _geocodificar_calle(zona_raw)
                if coords:
                    return coords
    # Búsqueda directa en texto (diccionario)
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
                    "rabo de toro", "chuleton", "callos madrilenos", "oreja"],

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
    "mexicano": "mexicana", "mexico": "mexicana", "mejico": "mexicana",
    "italiano": "italiana", "italia": "italiana",
    "japones": "japonesa", "japon": "japonesa",
    "indio": "india", "hindu": "india", "india": "india",
    "peruano": "peruana", "peru": "peruana",
    "espanol": "española", "espanola": "española",
    "asturiano": "asturiana", "asturias": "asturiana",
    "gallego": "gallega", "galicia": "gallega",
    "vasco": "vasca", "pais vasco": "vasca", "euskadi": "vasca",
    "frances": "francesa", "francia": "francesa",
    "griego": "griega", "grecia": "griega",
    "arabe": "arabe", "libanes": "arabe", "libano": "arabe",
    "venezolano": "venezolana", "venezuela": "venezolana",
    "colombiano": "colombiana", "colombia": "colombiana",
    "chino": "china",
    "tailandes": "tailandesa", "tailandia": "tailandesa", "thai": "tailandesa",
    "americano": "americana", "usa": "americana",
    "mediterraneo": "mediterranea",
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
    # Buscar sinónimos como palabras completas
    for sinonimo, cocina in SINONIMOS_COCINA.items():
        padded = " " + c + " "
        if (f" {sinonimo} " in padded or c == sinonimo
                or c.startswith(sinonimo + " ") or c.endswith(" " + sinonimo)):
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
    MIN_SEÑALES_PROPIAS = 2  # mínimo de platos propios para puntuar sin nombre_bonus

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
                        "patatas bravas", "gazpacho", "salmorejo"],
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
        "española":   ["paella", "chuleton", "jamón", "chorizo"],
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

    # Bonus por columna tipo_cocina si existe en el CSV
    tipo = _norm(str(row.get("tipo_cocina", "") or row.get("cocina", "") or ""))
    cocina_ascii = _norm(cocina)
    if tipo and (cocina_ascii in tipo or tipo in cocina_ascii):
        nombre_bonus = max(nombre_bonus, 8.0)

    # Contar platos propios encontrados
    señales_propias = 0
    score_propios = 0.0
    for plato in propios_def:
        plato_norm = _norm(plato)
        if plato_norm in fuente:
            señales_propias += 1
            score_propios += 3.0
            po = next(
                (p for p in platos_lista
                 if plato_norm in _norm(p["nombre"]) or _norm(p["nombre"]) in plato_norm),
                None
            )
            if po and po["menciones"] > 1:
                score_propios += math.log2(po["menciones"])

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
        "ninos":          "criterio_ninos",
        "mascotas":       "criterio_mascotas",
        "terraza":        "criterio_terraza",
        "vistas":         "criterio_vistas",
        "musica_directo": "criterio_musica_directo",
        "romantico":      "criterio_romantico",
    }
    if criterio in MAPA_CRITERIO_COL:
        return bool(row.get(MAPA_CRITERIO_COL[criterio], False))

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
    platos_destacados = [p["nombre"] for p in platos_lista]

    # Platos frecuencia: {"croquetas": 34, "pulpo": 21}
    platos_frecuencia = _parsear_platos_frecuencia(str(row.get("todos_platos", "") or row.get("top5_platos", "") or ""))

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
    criterio_ninos    = bool(row.get("criterio_ninos", False))
    criterio_mascotas = bool(row.get("criterio_mascotas", False))
    criterio_terraza  = bool(row.get("criterio_terraza", False))
    criterio_romantico = bool(row.get("criterio_romantico", False))
    criterio_vistas   = bool(row.get("criterio_vistas", False))

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

    # Calcular score base de calidad para todos
    df["_score_calidad"] = df.apply(_score_calidad, axis=1)

    # ── Score por texto/platos ─────────────────────────────────────
    if cocina:
        df["_score_match"] = df.apply(lambda r: _score_cocina(r, cocina), axis=1)
    else:
        df["_score_match"] = df.apply(lambda r: _score_texto(r, tokens), axis=1)

    # Buscar también por nombre del restaurante
    df["_score_nombre"] = df["nombre_display"].apply(
        lambda n: 3.0 if any(t in _norm(str(n)) for t in tokens if len(t) >= 3) else 0.0
    )

    # ── Score de distancia (si hay zona) ──────────────────────────
    if zona_coords:
        def _dist(row):
            lat, lon = row.get("latitud"), row.get("longitud")
            if pd.notna(lat) and pd.notna(lon):
                return _haversine(zona_coords, (float(lat), float(lon)))
            return 999.0
        df["_dist_zona"] = df.apply(_dist, axis=1)
        # Score distancia: 10 a 0 km → score 10 a 0
        max_dist = df["_dist_zona"].replace(999.0, pd.NA).dropna().max() or 10.0
        df["_score_dist"] = (1 - (df["_dist_zona"] / max_dist).clip(0, 1)) * 10
    else:
        df["_dist_zona"] = None
        df["_score_dist"] = 0.0

    # ── Score final combinado ──────────────────────────────────────
    if zona_coords:
        # Con zona: ordenar por distancia real, calidad solo como desempate
        # Se redondea la distancia a intervalos de 0.5 km para que la calidad
        # desempate entre restaurantes igualmente cercanos
        df["_dist_bucket"] = (df["_dist_zona"] / 0.5).apply(lambda x: round(x) if pd.notna(x) else 999)
        df["_score_final"] = (
            -df["_dist_bucket"] * 10 +          # primero los más cercanos
            df["_score_calidad"] * 1             # desempate por calidad dentro del mismo bucket
        )
    elif cocina or df["_score_match"].max() > 4:
        # Con match de texto/cocina: 50% match + 50% calidad
        df["_score_final"] = (
            (df["_score_match"] + df["_score_nombre"]).clip(0, 10) * 0.50 +
            df["_score_calidad"] * 0.50
        )
    else:
        # Sin match: solo calidad
        df["_score_final"] = df["_score_calidad"]

    # ── Filtrar por criterios detectados ──────────────────────────
    df_filtrado = df.copy()
    for criterio in criterios:
        mascara = df_filtrado.apply(lambda r: _pasa_criterio(r, criterio), axis=1)
        if mascara.sum() > 0:  # Solo filtrar si hay resultados
            df_filtrado = df_filtrado[mascara]

    # ── Si con filtros quedan pocos, usar sin filtros ──────────────
    if len(df_filtrado) < 3 and len(criterios) > 0:
        df_filtrado = df.copy()

    # ── Filtrar por score mínimo si hay cocina detectada ─────────────────────
    if cocina:
        # Solo incluir restaurantes que superaron el umbral de platos (score > 0)
        df_con_match = df_filtrado[df_filtrado["_score_match"] > 0]
        if len(df_con_match) >= 3:
            df_filtrado = df_con_match
        elif len(df_con_match) > 0:
            # Si hay menos de 3 con score > 0, los mostramos igualmente (son los únicos relevantes)
            df_filtrado = df_con_match

    # ── Filtrar por distancia máxima si hay zona ─────────────────
    if zona_coords and "_dist_zona" in df_filtrado.columns:
        cerca = df_filtrado[df_filtrado["_dist_zona"] <= 3.0]
        if len(cerca) >= 3:
            df_filtrado = cerca
        elif len(cerca) > 0:
            # Si hay pocos cerca de 3 km, ampliar a 5 km
            cerca5 = df_filtrado[df_filtrado["_dist_zona"] <= 5.0]
            if len(cerca5) >= 3:
                df_filtrado = cerca5

    # ── Ordenar y tomar top N ──────────────────────────────────────
    n_resultados = 6 if (cocina or tokens or criterios) else 8
    df_top = df_filtrado.sort_values("_score_final", ascending=False).head(n_resultados)

    restaurantes = []
    for _, row in df_top.iterrows():
        dist_km = None
        if zona_coords and pd.notna(row.get("_dist_zona")) and row["_dist_zona"] < 999.0:
            dist_km = row["_dist_zona"]
        restaurantes.append(_fila_a_restaurante(row, distancia_km=dist_km))

    meta = {
        "cocina": cocina,
        "zona": zona_coords,
        "criterios": criterios,
        "n_total": len(df_filtrado),
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
        return (
            "No he encontrado restaurantes que coincidan exactamente con tu búsqueda. "
            "Prueba con otros términos: tipo de cocina, plato concreto, zona de Madrid o criterios como terraza, niños, mascotas..."
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
            "romantico": "ambiente romántico", "ninos": "apto para niños",
            "mascotas": "que admiten mascotas", "terraza": "con terraza",
            "vistas": "con vistas", "musica_directo": "con música en directo",
            "tranquilo": "tranquilos", "precio_ok": "con buen precio",
            "muy_valorado": "muy valorados",
        }
        intro_parts.append(", ".join(etiquetas.get(c, c) for c in criterios))

    if intro_parts:
        intro = f"Aquí tienes mis recomendaciones de restaurantes con {' y '.join(intro_parts)}:"
    elif zona:
        intro = "Aquí tienes los restaurantes mejor valorados cerca de esa zona:"
    else:
        intro = f"Aquí tienes mis recomendaciones para «{consulta}»:"

    lineas = [intro, ""]

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
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    _inicializar()


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API Restaurantes Madrid — NLP local (nlptown)"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "restaurantes_cargados": len(df_global) if df_global is not None else 0,
        "backend": "nlptown + análisis NLP local",
    }


@app.post("/recomendar", response_model=RecomendacionResponse)
def endpoint_recomendar(request: ConsultaRequest):
    if not request.consulta.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
    if df_global is None:
        raise HTTPException(status_code=503, detail="Datos no cargados aún, espera un momento")

    try:
        restaurantes, meta = _buscar(request.consulta)
        respuesta = _generar_respuesta(request.consulta, restaurantes, meta)

        # Limpiar campos internos antes de devolver
        for r in restaurantes:
            r.pop("_pct_positivo", None)
            r.pop("_avg_estrellas", None)

        return RecomendacionResponse(
            respuesta=respuesta,
            proyecto="nlptown/bert + análisis NLP local",
            restaurantes=restaurantes,
            consulta_usuario=request.consulta,
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
