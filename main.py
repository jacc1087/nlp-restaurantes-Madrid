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
import urllib.request as _ureq
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
# GEMINI — NORMALIZADOR DE CONSULTAS
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_MODEL = "gemini-2.5-flash"

def _normalizar_consulta_gemini(consulta: str) -> str:
    """
    Usa Gemini para interpretar la consulta del usuario y normalizarla
    a términos que el motor de búsqueda entiende bien.
    Si Gemini no está disponible o falla, devuelve la consulta original.
    """
    if not _GEMINI_KEY or not consulta.strip():
        return consulta

    prompt = (
        "Normaliza esta búsqueda de restaurante en Madrid en UNA LÍNEA.\n\n"
        "EJEMPLOS (entrada → salida):\n"
        "quiero comer comida gallega → cocina gallega\n"
        "quiero un restaurante gallego → cocina gallega\n"
        "un buen restaurante italiano → cocina italiana\n"
        "me apetece un chino → cocina china\n"
        "me apetece un italiano → cocina italiana\n"
        "un japonés auténtico → cocina japonesa\n"
        "quiero comer croquetas → croquetas\n"
        "donde comer cachopo → cachopo\n"
        "croquetas cerca de malasaña → croquetas cerca de malasaña\n"
        "carne en lavapiés → carne en lavapiés\n"
        "algo romántico para cenar → romántico\n"
        "sitio para ir con niños → apto para niños\n"
        "algo con terraza → con terraza\n"
        "sitios conocidos de madrid → restaurantes famosos\n"
        "quiero cocido madrileño → cocido madrileño\n"
        "busco un restaurante castizo → cocina madrileña\n"
        "quiero callos a la madrileña → callos\n"
        "quiero gallinejas → gallinejas\n"
        "me apetece oreja a la plancha → oreja\n"
        "quiero un chuletón de vaca rubia gallega → chuletón\n"
        "quiero carne de vaca rubia gallega → asador\n"
        "me apetece un chuletón → chuletón\n"
        "quiero comer carne a la brasa → asador\n"
        "busco un buen asador → asador\n"
        "quiero un lechazo → lechazo\n"
        "quiero cordero asado → asador\n"
        "quiero comer marisco → marisco\n"
        "una marisquería → marisco\n"
        "me apetecen unas ostras → marisco\n"
        "quiero percebes → marisco\n"
        "quiero una paella → paella\n"
        "vamos a una arrocería → arroceria\n"
        "quiero un arroz con bogavante → arroz con bogavante\n"
        "me apetece una fideuá → fideuá\n"
        "quiero una hamburguesa → hamburguesa\n"
        "busco una hamburguesería → hamburguesa\n"
        "quiero tapas → tapas\n"
        "busco un sitio de tapas → tapas\n"
        "quiero sushi → cocina japonesa\n"
        "busco un sitio de ramen → ramen\n"
        "quiero curry → cocina india\n"
        "quiero tacos → cocina mexicana\n"
        "busco un buen ceviche → ceviche\n"
        "quiero fusión → cocina fusión\n"
        "alta cocina en madrid → cocina fusión\n\n"
        "REGLAS:\n"
        "- Cocina/gastronomía de un país o región: SIEMPRE devuelve 'cocina X' (cocina gallega, cocina italiana...)\n"
        "- Tipo de restaurante (asador, marisquería, arrocería, hamburguesería): devuelve solo ese término\n"
        "- Plato concreto: solo el nombre del plato\n"
        "- Si hay zona: consérvala\n"
        "- NO añadas nada que no esté en la consulta\n\n"
        f"Consulta: {consulta!r}\n\n"
        "Responde SOLO con la versión normalizada, sin comillas."
    )

    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 100, "stopSequences": []},
            "safetySettings": [],
        }
        data = json.dumps(payload).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent?key={_GEMINI_KEY}"
        req = _ureq.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        resp = _ureq.urlopen(req, timeout=10)
        raw = resp.read().decode("utf-8")
        result = json.loads(raw)
        candidates = result.get("candidates", [])
        if not candidates:
            print(f"  [Gemini] sin candidatos: {raw[:200]}")
            return consulta
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            print(f"  [Gemini] sin parts: {str(candidates[0])[:200]}")
            return consulta
        normalizada = "".join(p.get("text", "") for p in parts).strip()
        # Seguridad: si la respuesta es muy larga o rara, usar la original
        if normalizada and len(normalizada) < 200 and len(normalizada) > 2:
            print(f"  [Gemini] '{consulta}' → '{normalizada}'")
            return normalizada
    except Exception as e:
        print(f"  [Gemini normalizer] error: {e}")

    return consulta


def _generar_resumen_gemini(row) -> str:
    """
    Usa Gemini para generar un párrafo de presentación del restaurante
    basado exclusivamente en datos reales extraídos de sus reseñas.
    Si Gemini no está disponible o falla, devuelve un resumen automático básico.
    """
    if not _GEMINI_KEY:
        return ""

    nombre    = str(row.get("nombre", "") or "")
    n         = int(row.get("n_resenas", 0) or 0)
    pct_pos   = float(row.get("pct_positivo", 0) or 0)
    val       = float(row.get("valoracion_google", 0) or 0)
    todos_platos = str(row.get("todos_platos", "") or "")
    servicio_frases = str(row.get("servicio_frases", "") or "")
    personal  = str(row.get("personal_destacado", "") or "")
    terminos  = str(row.get("terminos_tfidf", "") or "")

    # Dimensiones NLP
    comida_avg   = float(row.get("comida_avg_stars", 0) or 0)
    servicio_avg = float(row.get("servicio_avg_stars", 0) or 0)
    ambiente_avg = float(row.get("ambiente_avg_stars", 0) or 0)
    precio_avg   = float(row.get("precio_avg_stars", 0) or 0)

    # Criterios detectados
    criterios = []
    if row.get("criterio_terraza"):   criterios.append("terraza")
    if row.get("criterio_romantico"): criterios.append("ambiente romántico")
    if row.get("criterio_ninos"):     criterios.append("apto para niños")
    if row.get("criterio_vistas"):    criterios.append("con vistas")
    if row.get("criterio_mascotas"):  criterios.append("admite mascotas")

    # Frases de servicio — máx 3 fragmentos
    frases = ""
    if servicio_frases and servicio_frases != "nan":
        fragmentos = [f.strip() for f in servicio_frases.split("|") if len(f.strip()) > 20][:3]
        frases = " | ".join(fragmentos)

    dim_texto = []
    if comida_avg   > 0: dim_texto.append(f"comida {comida_avg:.1f}/5")
    if servicio_avg > 0: dim_texto.append(f"servicio {servicio_avg:.1f}/5")
    if ambiente_avg > 0: dim_texto.append(f"ambiente {ambiente_avg:.1f}/5")
    if precio_avg   > 0: dim_texto.append(f"precio {precio_avg:.1f}/5")

    prompt = (
        f"Eres un experto en gastronomía madrileña. Escribe UN párrafo de presentación "
        f"atractivo y natural para el restaurante '{nombre}', dirigido a alguien que busca "
        f"dónde comer en Madrid. Usa SOLO los datos que te doy, sin inventar nada.\n\n"
        f"DATOS DEL RESTAURANTE (extraídos de {n} reseñas reales de Google):\n"
        f"- Valoración Google: {val}/5 ({pct_pos:.0f}% de reseñas positivas)\n"
        f"- Platos más mencionados en reseñas: {todos_platos[:200]}\n"
    )
    if dim_texto:
        prompt += f"- Puntuación por dimensión: {', '.join(dim_texto)}\n"
    if personal and personal != "nan":
        prompt += f"- Personal destacado por clientes: {personal[:150]}\n"
    if frases:
        prompt += f"- Fragmentos literales de reseñas sobre el servicio: {frases[:400]}\n"
    if terminos and terminos != "nan":
        prompt += f"- Términos más característicos según los clientes: {terminos}\n"
    if criterios:
        prompt += f"- Características destacadas: {', '.join(criterios)}\n"

    prompt += (
        f"\nEscribe el párrafo en español, máximo 3 frases. "
        f"Menciona los platos estrella, la valoración general y algo del ambiente o servicio si los datos lo justifican. "
        f"Tono cálido y directo. NO uses frases como 'según las reseñas' o 'los clientes dicen' — "
        f"escríbelo como si fuera una presentación editorial, pero basada en los datos reales."
    )

    try:
        import json as _json, urllib.request as _ureq
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 200},
            "safetySettings": [],
        }
        data = _json.dumps(payload).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent?key={_GEMINI_KEY}"
        req = _ureq.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        resp = _ureq.urlopen(req, timeout=15)
        result = _json.loads(resp.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texto = "".join(p.get("text", "") for p in parts).strip()
            if texto and len(texto) > 20:
                return texto
    except Exception as e:
        print(f"  [Gemini resumen] error para {nombre}: {e}")

    return ""


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
            # 3. Geocodificar como calle de Madrid — solo si parece una calle real
            # Evitar geocodificar frases como "comer buen pulpo", "restaurante italiano"
            # Palabras que nunca son zonas geográficas
            PALABRAS_NO_ZONA = {
                'comer','buen','buena','buenos','buenas','pulpo','croquetas',
                'paella','pizza','sushi','ramen','cachopo','fabada','ceviche',
                'tapas','pasta','arroz','carne','pescado','marisco','mariscos',
                'restaurante','cocina','comida','sitio','lugar','cerca','donde',
                'italiano','japones','peruano','gallego','vasco','mexicano',
                'italiano','francesa','griega','india','arabe','venezolana',
                'bueno','rico','rica','especial','tipico','tipica','autentico',
                'moderno','moderna','tradicional','casero','casera',
            }
            # También descartar si el texto capturado es un plato conocido
            from itertools import chain
            todos_platos_conocidos = set(chain.from_iterable(
                v for v in PLATOS_COCINA.values()
            )) if 'PLATOS_COCINA' in dir() else set()
            palabras_zona = set(zona_raw.split())
            es_plato = any(zona_raw in p or p in zona_raw
                          for p in todos_platos_conocidos) if todos_platos_conocidos else False
            if len(zona_raw) >= 4 and not palabras_zona.intersection(PALABRAS_NO_ZONA) and not es_plato:
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
    # Mexicana
    "mexicano": "mexicana", "mexicana": "mexicana", "mexico": "mexicana", "mejico": "mexicana",
    "tacos": "mexicana", "taqueria": "mexicana", "burrito": "mexicana", "burritos": "mexicana",
    "fajitas": "mexicana", "guacamole": "mexicana", "quesadilla": "mexicana",
    # Italiana
    "italiano": "italiana", "italiana": "italiana", "italia": "italiana",
    "pasta": "italiana", "pizza": "italiana", "pizzeria": "italiana",
    "trattoria": "italiana", "carbonara": "italiana", "lasana": "italiana", "lasaña": "italiana",
    "risotto": "italiana", "tiramisu": "italiana", "tiramisú": "italiana",
    # Japonesa
    "japones": "japonesa", "japonesa": "japonesa", "japon": "japonesa",
    "sushi": "japonesa", "ramen": "japonesa", "gyozas": "japonesa", "tempura": "japonesa",
    "sashimi": "japonesa", "udon": "japonesa", "japonés": "japonesa",
    # India
    "indio": "india", "india": "india", "hindu": "india",
    "curry": "india", "tikka": "india", "tikka masala": "india",
    "naan": "india", "biryani": "india", "tandoori": "india",
    # Peruana
    "peruano": "peruana", "peruana": "peruana", "peru": "peruana",
    "ceviche": "peruana", "lomo saltado": "peruana", "tiradito": "peruana",
    "causa": "peruana", "anticuchos": "peruana",
    # Madrileña
    "madrileno": "madrileña", "madrilena": "madrileña",
    "madrileño": "madrileña", "madrileña": "madrileña",
    "cocido": "madrileña", "cocido madrileño": "madrileña", "callos": "madrileña",
    "oreja": "madrileña", "oreja a la plancha": "madrileña",
    "gallinejas": "madrileña", "entresijos": "madrileña",
    "caracoles": "madrileña", "sesos": "madrileña",
    "manitas": "madrileña", "manitas de cerdo": "madrileña",
    "cocina castiza": "madrileña", "castizo": "madrileña", "castiza": "madrileña",
    "taberna castiza": "madrileña", "cocina madrileña": "madrileña",
    # Asturiana
    "asturiano": "asturiana", "asturiana": "asturiana", "asturias": "asturiana",
    "cachopo": "asturiana", "fabada": "asturiana", "sidra": "asturiana",
    # Gallega
    "gallego": "gallega", "galicia": "gallega",
    "cocina gallega": "gallega", "comida gallega": "gallega",
    "restaurante gallego": "gallega", "pulpo gallego": "gallega",
    "empanada gallega": "gallega", "percebes": "gallega",
    # Vasca
    "vasco": "vasca", "vasca": "vasca", "pais vasco": "vasca", "euskadi": "vasca",
    "pintxos": "vasca", "pintxo": "vasca", "txangurro": "vasca", "marmitako": "vasca",
    # Francesa
    "frances": "francesa", "francesa": "francesa", "francia": "francesa",
    # Griega
    "griego": "griega", "griega": "griega", "grecia": "griega",
    "gyros": "griega", "souvlaki": "griega", "moussaka": "griega",
    # Árabe
    "arabe": "arabe", "libanes": "arabe", "libano": "arabe",
    "falafel": "arabe", "shawarma": "arabe", "hummus": "arabe", "kebab": "arabe",
    # Venezolana
    "venezolano": "venezolana", "venezolana": "venezolana", "venezuela": "venezolana",
    "arepa": "venezolana", "arepas": "venezolana", "tequeños": "venezolana",
    # Colombiana
    "colombiano": "colombiana", "colombiana": "colombiana", "colombia": "colombiana",
    # China
    "chino": "china", "china": "china",
    "dim sum": "china", "wonton": "china", "dumplings": "china",
    # Tailandesa
    "tailandes": "tailandesa", "tailandesa": "tailandesa", "tailandia": "tailandesa",
    "thai": "tailandesa", "pad thai": "tailandesa", "tom yum": "tailandesa",
    # Americana / Hamburguesería
    "americano": "americana", "americana": "americana", "usa": "americana",
    "hamburguesa": "americana", "hamburguesas": "americana", "burger": "americana",
    "smash burger": "americana", "hamburgueseria": "americana",
    # Argentina / Asador / Carne
    "argentino": "argentina", "argentina_cocina": "argentina",
    "asador": "asador", "asadores": "asador",
    "chuleton": "asador", "chuletón": "asador",
    "lechazo": "asador", "cordero asado": "asador", "cordero": "asador",
    "carne a la brasa": "asador", "parrilla": "asador",
    "entrana": "asador", "entraña": "asador",
    "buey": "asador", "buey a la brasa": "asador",
    "lomo alto": "asador",
    # Marisquería
    "marisco": "marisqueria", "mariscos": "marisqueria",
    "marisqueria": "marisqueria", "marisquería": "marisqueria",
    "percebes": "marisqueria", "ostras": "marisqueria", "vieiras": "marisqueria",
    "bogavante": "marisqueria", "langosta": "marisqueria", "carabineros": "marisqueria",
    "navajas": "marisqueria", "zamburinas": "marisqueria", "zamburiñas": "marisqueria",
    "marisco fresco": "marisqueria", "pescado y marisco": "marisqueria",
    # Arrocería / Paella
    "arroceria": "arroceria", "arrocería": "arroceria",
    "paella": "arroceria", "paellas": "arroceria",
    "arroz": "arroceria", "arroces": "arroceria",
    "arroz con bogavante": "arroceria", "arroz negro": "arroceria",
    "fideuá": "arroceria", "fideua": "arroceria",
    # Andaluza
    "andaluz": "andaluza", "andaluza": "andaluza", "andalucia": "andaluza",
    "andalucía": "andaluza", "sevilla": "andaluza", "sevillano": "andaluza",
    "gaditano": "andaluza", "gaditana": "andaluza", "cadiz": "andaluza",
    "jerez": "andaluza", "malaga": "andaluza", "cordoba": "andaluza",
    "pescaito": "andaluza", "pescaito frito": "andaluza",
    "tortillitas de camarones": "andaluza", "ajoblanco": "andaluza",
    "berenjenas con miel": "andaluza", "cola de toro": "andaluza",
    "salmorejo": "andaluza",
    # Catalana
    "catalan": "catalana", "catalana": "catalana", "cataluna": "catalana",
    "cataluña": "catalana", "barcelona": "catalana",
    # Taberna / Tapas
    "taberna": "taberna", "tapas": "taberna", "de tapas": "taberna",
    "bar de tapas": "taberna", "tasca": "taberna", "tasquita": "taberna",
    "vermut": "taberna", "vermú": "taberna", "vermuth": "taberna",
    "pinchos": "taberna",
    # Mediterránea
    "mediterraneo": "mediterranea", "mediterranea": "mediterranea",
    # Fusión / Alta cocina
    "fusion": "fusion", "fusión": "fusion",
    "alta cocina": "fusion", "cocina creativa": "fusion",
    "estrella michelin": "fusion", "michelin": "fusion",
    "gastronomico": "fusion", "gastronómico": "fusion",
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


# IDs de restaurantes icónicos/famosos de Madrid — selección manual por notoriedad
# Combina: estrellas Michelin, alta popularidad en Google, referentes culturales
RESTAURANTES_ICONICOS = {
    70,   # DiverXO
    40,   # DSTAGE
    128,  # Coque
    130,  # StreetXO
    132,  # Corral de la Morería
    125,  # Dans Le Noir ?
    95,   # Bacira
    75,   # sr.Ito
    62,   # La Tasqueria de Javi Estevez
    156,  # Juana La Loca
    24,   # Rosi La Loca
    74,   # Bestial By Rosi La Loca
    85,   # Angelita Madrid
    131,  # La Castela
    103,  # Casa Benigna
    1,    # Los Montes de Galicia
    9,    # Inclán Brutal Bar
    6,    # Tabernas El Sur
    86,   # Casa Lucas
    154,  # Dantxari
    155,  # Casa Toni
}

PALABRAS_FAMOSOS = [
    "famoso", "famosa", "famosos", "famosas",
    "conocido", "conocida", "conocidos", "conocidas",
    "popular", "populares",
    "iconico", "iconica", "iconicos",
    "referente", "referentes",
    "top madrid", "lo mejor de madrid", "imprescindible", "imprescindibles",
    "estrella michelin", "michelin", "con estrella",
    "de moda", "trendy", "instagram",
    "recomendado", "muy recomendado",
    "especial", "unico", "unica",
    "gastro", "gastronomico", "gastronomica",
    "experiencia", "experiencia unica",
    "must", "must do", "hay que ir",
]

def _detectar_famosos(consulta: str) -> bool:
    """Detecta si el usuario busca restaurantes famosos/icónicos/conocidos."""
    c = _norm(consulta)
    return any(p in c for p in PALABRAS_FAMOSOS)


CATEGORIAS_ESPECIALES = {'asador', 'marisqueria', 'arroceria', 'hamburgueseria',
                         'fusion', 'taberna', 'italiano', 'japones', 'indio',
                         'mexicano', 'peruano', 'venezolano', 'argentino',
                         'tailandes', 'arabe'}

def _detectar_cocina(consulta: str) -> Optional[str]:
    c = _norm(consulta)
    padded = " " + c + " "
    # Prefijos que preceden al tipo de cocina
    for prefijo in ["comida ", "restaurante ", "cocina ", "cuisine "]:
        for sinonimo, cocina in SINONIMOS_COCINA.items():
            if prefijo + sinonimo in c:
                return cocina
    # Buscar sinónimos como palabras completas
    for sinonimo, cocina in SINONIMOS_COCINA.items():
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
    Score de afinidad con una cocina/categoría.
    - Si es una categoría especial (asador, marisqueria, etc.): busca en categoria_carta.
    - Si es una cocina de país: usa los platos como antes.
    """
    # Mapa de valor detectado → valor en categoria_carta
    MAPA_CATEGORIA = {
        'asador':       'asador',
        'marisqueria':  'marisqueria',
        'arroceria':    'arroceria',
        'hamburgueseria': 'hamburgueseria',
        'fusion':       'fusion',
        'taberna':      'taberna',
        'italiana':     'italiano',
        'japonesa':     'japones',
        'india':        'indio',
        'mexicana':     'mexicano',
        'peruana':      'peruano',
        'venezolana':   'venezolano',
        'argentina':    'argentino',
        'tailandesa':   'tailandes',
        'arabe':        'arabe',
        'americana':    'hamburgueseria',
    }

    categoria_buscada = MAPA_CATEGORIA.get(cocina)
    categoria_restaurante = _norm(str(row.get('categoria_carta', '') or ''))

    if categoria_buscada:
        # Si el restaurante tiene categoria_carta y coincide → score alto
        if categoria_restaurante == _norm(categoria_buscada):
            # Bonus por calidad dentro de la categoría
            return 10.0 + _score_calidad(row) * 0.5
        # Si no coincide categoria_carta, score 0 — no mezclar categorías
        return 0.0

    # Cocinas sin categoría especial (gallega, vasca, madrileña, etc.) → lógica original por platos
    # PERO primero comprobar cocina_detectada — es la señal más fiable
    cocina_rest = _norm(str(row.get('cocina_detectada', '') or ''))
    if cocina_rest == _norm(cocina):
        return 10.0 + _score_calidad(row) * 0.5

    MIN_PLATOS_COCINA = 3

    PLATOS_COCINA = {
        "gallega":     ["pulpo", "empanada", "percebes", "navajas", "vieiras", "berberechos",
                        "zamburinas", "caldo gallego", "lacon", "grelos", "padron", "filloas",
                        "tetilla", "pote gallego", "zorza", "ribeiro", "albarino"],
        "italiana":    ["carbonara", "cacio e pepe", "amatriciana", "ossobuco", "panna cotta",
                        "risotto", "tagliatelle", "pappardelle", "gnocchi", "cannoli", "arancini",
                        "burrata", "stracciatella", "tiramisu", "lasana", "lasaña", "pasta", "pizza",
                        "bruschetta", "focaccia", "ribollita"],
        "peruana":     ["lomo saltado", "lomo salteado", "causa", "anticuchos", "chaufa",
                        "aji de gallina", "leche de tigre", "tiradito", "ceviche",
                        "pachamanquero", "chicharron peruano"],
        "japonesa":    ["ramen", "sushi", "sashimi", "gyozas", "tempura", "udon", "mochi",
                        "katsu", "takoyaki", "tonkatsu", "nigiri", "yakitori", "edamame",
                        "miso", "soba"],
        "vasca":       ["pintxos", "pintxo", "gilda", "bacalao pil pil", "txangurro", "marmitako",
                        "kokotxas", "txakoli", "chipirones en su tinta", "merluza en salsa verde",
                        "bacalao", "merluza", "anchoas"],
        "asturiana":   ["cachopo", "cachopu", "fabada", "fabada asturiana", "oricios",
                        "pote asturiano", "cabrales", "casadielles", "sidra",
                        "verdinas", "compango", "frixuelos", "tortos"],
        "india":       ["tikka masala", "biryani", "naan", "samosa", "korma", "dal", "tandoori",
                        "chapati", "pakora", "butter chicken", "palak paneer", "chana masala",
                        "lassi", "dosa", "saag"],
        "venezolana":  ["arepa", "arepas", "pabellon criollo", "cachapa", "hallaca", "pernil",
                        "caraotas", "tequeños", "mandocas", "chicha"],
        "mexicana":    ["burrito", "quesadilla", "fajitas", "guacamole", "enchilada", "pozole",
                        "carnitas", "mole", "tacos", "chilaquiles", "chile relleno", "tamales",
                        "tostadas", "nachos"],
        "griega":      ["gyros", "souvlaki", "moussaka", "spanakopita", "tzatziki", "baklava",
                        "dolmades", "kleftiko", "taramasalata", "hummus"],
        "francesa":    ["foie gras", "confit de pato", "bouillabaisse", "coq au vin", "escargots",
                        "crepe", "souffle", "cassoulet", "magret", "ratatouille", "tartare"],
        "arabe":       ["shawarma", "falafel", "tabule", "baba ganoush", "labneh", "shakshuka",
                        "kibbeh", "fattoush", "couscous", "pita", "hummus", "kebab"],
        "colombiana":  ["bandeja paisa", "ajiaco", "sancocho", "changua", "lechona",
                        "tamales colombianos", "arepa"],
        "china":       ["dim sum", "wonton", "pato pekin", "chow mein", "baozi", "mapo tofu",
                        "pato laqueado", "dumplings", "spring roll"],
        "tailandesa":  ["pad thai", "tom yum", "massaman", "curry verde", "curry rojo", "satay",
                        "som tam", "larb", "mango sticky rice", "khao pad"],
        "americana":   ["smash burger", "pulled pork", "costillas bbq", "mac and cheese",
                        "chicken wings", "brisket", "coleslaw", "brownie", "costillar"],
        "madrileña":   ["cocido madrileño", "cocido madrileno", "cocido", "callos", "callos madrileños",
                        "bocadillo calamares", "soldaditos pavia", "migas"],
    }

    platos_def = PLATOS_COCINA.get(cocina, [])
    if not platos_def:
        return 0.0

    fuente = _norm(
        str(row.get("todos_platos", "") or "") + " " +
        str(row.get("terminos_tfidf", "") or "")
    )
    platos_lista = _parsear_platos_str(str(row.get("todos_platos", "") or ""))

    matches = []
    score = 0.0
    for plato in platos_def:
        if _norm(plato) in fuente:
            po = next(
                (p for p in platos_lista
                 if _norm(plato) in _norm(p["nombre"]) or _norm(p["nombre"]) in _norm(plato)),
                None
            )
            menciones = po["menciones"] if po else 1
            matches.append({"plato": plato, "menciones": menciones})
            score += 2.0
            if menciones > 1:
                score += math.log2(menciones)

    if len(matches) < MIN_PLATOS_COCINA:
        return 0.0

    con_menciones = sum(1 for m in matches if m["menciones"] > 1)
    plato_estrella = any(m["menciones"] > 8 for m in matches)
    if con_menciones < 2 and not plato_estrella:
        return 0.0

    total_menciones_restaurante = sum(p["menciones"] for p in platos_lista)
    menciones_cocina = sum(m["menciones"] for m in matches)
    if total_menciones_restaurante > 0:
        proporcion = menciones_cocina / total_menciones_restaurante
        if proporcion < 0.35:
            return 0.0

    return round(score, 2)


def _score_texto(row: pd.Series, tokens_query: list) -> float:
    """
    Score de coincidencia para búsquedas de plato específico.
    Si el token coincide con un plato en todos_platos, exige MIN_MENCIONES_PLATO
    menciones para puntuar — evita devolver restaurantes donde ese plato apareció
    una sola vez de pasada en una reseña.
    """
    MIN_MENCIONES_PLATO = 3  # mínimo de menciones para considerar el plato como real

    platos_lista = _parsear_platos_str(str(row.get("todos_platos", "") or ""))
    nombre_display = _norm(str(row.get("nombre_display", "") or ""))

    score = 0.0
    for qt in tokens_query:
        if len(qt) < 3:
            continue

        # Buscar el token en los platos del restaurante
        po = next(
            (p for p in platos_lista
             if qt in _norm(p["nombre"]) or _norm(p["nombre"]) in qt),
            None
        )

        if po:
            # Coincide con un plato: exigir mínimo de menciones
            if po["menciones"] >= MIN_MENCIONES_PLATO:
                score += 2.0 + math.log2(po["menciones"])
            # Con pocas menciones no puntúa — el plato no es representativo
        else:
            # No está en todos_platos: buscar en terminos_tfidf o nombre
            fuente_sec = _norm(
                str(row.get("terminos_tfidf", "") or "") + " " + nombre_display
            )
            if qt in fuente_sec:
                score += 1.0  # coincidencia débil (término genérico o nombre)

    return score


def _score_calidad(row: pd.Series) -> float:
    """
    Score de calidad combinado 0-10.

    Pesos base:
      35% — valoración Google bayesiana (ponderada por nº de votaciones)
      35% — % reseñas positivas (nlptown)
      30% — avg_estrellas_modelo (nlptown, 1-5 → 0-10)

    La valoración Google se pondera bayesianamente: un restaurante con pocas
    votaciones se acerca a la media global (prior = 4.3, m = 500 votos).
    Esto evita que un 4.9 con 200 votos supere a un 4.7 con 10.000.
    """
    # ── Valoración Google bayesiana ───────────────────────────────
    PRIOR_MEDIA = 4.3   # media global estimada del conjunto
    PRIOR_VOTOS = 500   # peso del prior: equivale a N votos "ficticios" en la media

    val_raw  = float(row.get("valoracion_display", 0) or 0)
    votos    = float(row.get("votaciones", 0) or 0)

    if votos > 0 and val_raw > 0:
        val_bay = (PRIOR_VOTOS * PRIOR_MEDIA + votos * val_raw) / (PRIOR_VOTOS + votos)
    else:
        val_bay = PRIOR_MEDIA  # sin votos → asignar media

    val_norm = (val_bay / 5.0) * 10

    # ── NLP ───────────────────────────────────────────────────────
    pct_pos  = float(row.get("pct_positivo", 0) or 0)   # 0-100
    avg_est  = float(row.get("avg_estrellas_modelo", 3) or 3)
    avg_norm = ((avg_est - 1) / 4.0) * 10               # 1-5 → 0-10

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

    # Generar resumen con Gemini basado en datos reales de reseñas
    resumen = _generar_resumen_gemini(row)
    if not resumen:
        # Fallback si Gemini no está disponible
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
        # Personal destacado y valoración de servicio
        "personal_destacado":           "" if str(row.get("personal_destacado", "") or "").strip() in ("", "nan", "NaN", "None") else str(row.get("personal_destacado", "")),
        "servicio_frases":              str(row.get("servicio_frases", "") or ""),
        "servicio_pos":                 float(row.get("servicio_pos", 0) or 0),
        "servicio_menciones":           float(row.get("servicio_menciones", 0) or 0),
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
    # Filtrar tokens que no aportan información de búsqueda
    TOKENS_BASURA = {
        'restaurante','restaurantes','donde','comer','quiero','busco','buscar',
        'buen','buena','buenos','buenas','mejor','mejores','rico','rica',
        'sitio','lugar','alguno','alguna','hay','para','que','con','una','uno',
        'tipo','clase','algo','cerca','aqui','alli','puedo','pueda','ir','ver',
        'necesito','tengo','ganas','apetece','recomiendas','recomienda',
    }
    tokens = [t for t in re.split(r'[\s,.()/\-]+', consulta_norm)
              if len(t) >= 3 and t not in TOKENS_BASURA]

    # Detectar intención
    cocina = _detectar_cocina(consulta)
    zona_coords = _detectar_zona(consulta)
    criterios = _detectar_criterios(consulta)
    famosos = _detectar_famosos(consulta)

    # Si no se detecta nada concreto, usar Gemini para interpretar
    tokens_tmp = [t for t in __import__('re').split(r'[\s,.()/\-]+', consulta_norm)
                  if len(t) >= 3 and t not in TOKENS_BASURA]
    if not cocina and not criterios and not famosos and not tokens_tmp:
        consulta_norm_gemini = _normalizar_consulta_gemini(consulta)
        if consulta_norm_gemini != consulta:
            consulta = consulta_norm_gemini
            consulta_norm = _norm(consulta)
            cocina = _detectar_cocina(consulta)
            zona_coords = zona_coords or _detectar_zona(consulta)
            criterios = _detectar_criterios(consulta)
            famosos = _detectar_famosos(consulta)

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

    # ── Filtrar primero, luego calcular score final ────────────────
    df_filtrado = df.copy()

    # FILTRO POR FAMOSOS: si busca sitios conocidos/icónicos, filtrar por lista curada
    if famosos and not cocina:
        # IDs que siempre aparecen primero por ser landmarks absolutos de Madrid
        ICONICOS_PRIMERO = [70, 40, 128, 130, 132, 125]  # DiverXO, DSTAGE, Coque, StreetXO, Corral Morería, Dans Le Noir
        ids_int = df["id_restaurante"].astype(int)
        mask = ids_int.isin(RESTAURANTES_ICONICOS)
        if mask.sum() > 0:
            df_famosos = df[mask].copy()
            df_famosos["_score_final"] = df_famosos.apply(_score_calidad, axis=1)
            # Los landmarks fijos van siempre primero con score extra
            df_famosos["_es_landmark"] = ids_int[mask].isin(ICONICOS_PRIMERO).astype(int) * 100
            df_famosos["_score_final"] = df_famosos["_score_final"] + df_famosos["_es_landmark"]
            df_top = df_famosos.sort_values("_score_final", ascending=False).head(8)
            restaurantes = [_fila_a_restaurante(row) for _, row in df_top.iterrows()]
            meta = {"cocina": None, "zona": zona_coords, "criterios": criterios, "n_total": len(df_famosos), "famosos": True, "tokens": tokens}
            return restaurantes, meta

    # FILTRO ESTRICTO POR COCINA: aplicar ANTES de cualquier otra lógica.
    if cocina:
        df_filtrado = df_filtrado[df_filtrado["_score_match"] > 0]
        if df_filtrado.empty:
            return [], {"cocina": cocina, "zona": zona_coords, "criterios": criterios, "n_total": 0}

    # FILTRO PLATO + ZONA: si el usuario busca un plato concreto cerca de una zona,
    # filtrar primero por los que tienen ese plato, luego ordenar por distancia.
    # Evita devolver los restaurantes más cercanos que no tienen el plato.
    if not cocina and zona_coords and tokens:
        max_match = df["_score_match"].max()
        if max_match > 0:
            df_con_plato = df_filtrado[df_filtrado["_score_match"] > 0]
            if not df_con_plato.empty:
                df_filtrado = df_con_plato
            else:
                return [], {"cocina": None, "zona": zona_coords, "criterios": criterios, "n_total": 0, "tokens": tokens}

    # Filtrar por criterios (terraza, niños, etc.) — solo si hay resultados previos
    for criterio in criterios:
        mascara = df_filtrado.apply(lambda r: _pasa_criterio(r, criterio), axis=1)
        if mascara.sum() > 0:
            df_filtrado = df_filtrado[mascara]

    # Filtrado por texto específico (sin cocina, sin zona)
    if not cocina and not zona_coords:
        max_match = df["_score_match"].max()
        if max_match > 0:
            # Hay restaurantes que coinciden con el plato/término buscado
            df_filtrado = df_filtrado[df_filtrado["_score_match"] > 0]
            if df_filtrado.empty:
                # Ninguno pasa el umbral de menciones → devolver vacío, no relleno genérico
                return [], {"cocina": None, "zona": zona_coords, "criterios": criterios, "n_total": 0, "tokens": tokens}
        elif tokens:
            # Se buscó algo específico pero ningún restaurante lo tiene con suficientes menciones
            return [], {"cocina": None, "zona": zona_coords, "criterios": criterios, "n_total": 0, "tokens": tokens}

    # ── Score final solo sobre los restaurantes filtrados ──────────
    if zona_coords and cocina:
        # Cocina + zona: primero se exige cocina (ya filtrado arriba),
        # luego ordenar por proximidad y dentro de cada distancia por calidad
        df_filtrado["_dist_bucket"] = (df_filtrado["_dist_zona"] / 0.5).apply(lambda x: round(x) if pd.notna(x) else 999)
        df_filtrado["_score_final"] = (
            -df_filtrado["_dist_bucket"] * 10 +
            df_filtrado["_score_calidad"] * 1
        )
    elif zona_coords:
        df_filtrado["_dist_bucket"] = (df_filtrado["_dist_zona"] / 0.5).apply(lambda x: round(x) if pd.notna(x) else 999)
        df_filtrado["_score_final"] = (
            -df_filtrado["_dist_bucket"] * 10 +
            df_filtrado["_score_calidad"] * 1
        )
    elif cocina:
        # Solo cocina: ordenar por calidad (el match ya garantiza que son de esa cocina)
        df_filtrado["_score_final"] = (
            df_filtrado["_score_match"].clip(0, 10) * 0.40 +
            df_filtrado["_score_calidad"] * 0.60
        )
    elif df_filtrado["_score_match"].max() > 4:
        df_filtrado["_score_final"] = (
            (df_filtrado["_score_match"] + df_filtrado["_score_nombre"]).clip(0, 10) * 0.50 +
            df_filtrado["_score_calidad"] * 0.50
        )
    else:
        df_filtrado["_score_final"] = df_filtrado["_score_calidad"]

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
    # Si hay búsqueda específica (cocina o plato), devolver solo los que pasan el filtro
    # sin rellenar con otros — aunque sea 1 solo resultado.
    # En búsqueda genérica (sin filtros), limitar a 8 para no saturar.
    if cocina or (tokens and df_filtrado["_score_match"].max() > 0):
        n_resultados = min(len(df_filtrado), 8)  # máximo 8, pero sin relleno
    elif criterios:
        n_resultados = 8
    else:
        n_resultados = 8
    df_top = df_filtrado.sort_values("_score_final", ascending=False).head(n_resultados)

    restaurantes = []
    for _, row in df_top.iterrows():
        dist_km = None
        if zona_coords and pd.notna(row.get("_dist_zona")) and row["_dist_zona"] < 999.0:
            dist_km = row["_dist_zona"]
        r = _fila_a_restaurante(row, distancia_km=dist_km)
        r["tokens"] = tokens  # tokens de la búsqueda para que el frontend destaque el plato
        restaurantes.append(r)

    meta = {
        "cocina": cocina,
        "zona": zona_coords,
        "criterios": criterios,
        "n_total": len(df_filtrado),
        "tokens": tokens,
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
        tokens_buscados = meta.get("tokens", [])
        if cocina_vacia:
            return (
                f"No hay ningún restaurante en la base de datos con suficientes platos de cocina {cocina_vacia} "
                f"(mínimo 3 platos identificados en reseñas). No te devuelvo alternativas genéricas."
            )
        if tokens_buscados:
            plato = " ".join(tokens_buscados)
            return (
                f"No he encontrado ningún restaurante donde «{plato}» aparezca con suficientes menciones en reseñas. "
                f"Puede que ese plato no esté bien representado en los datos o que ningún restaurante lo destaque."
            )
        return (
            "No he encontrado restaurantes que coincidan exactamente con tu búsqueda. "
            "Prueba con otros términos: tipo de cocina, plato concreto, zona de Madrid o criterios como terraza, niños, mascotas..."
        )

    cocina    = meta.get("cocina")
    zona      = meta.get("zona")
    criterios = meta.get("criterios", [])
    famosos   = meta.get("famosos", False)

    # Intro contextual
    intro_parts = []
    if famosos:
        intro = "Aquí tienes algunos de los restaurantes más conocidos e icónicos de Madrid:"
        lineas = [intro, ""]
        # reutilizar el mismo bucle de formato
        for r in restaurantes:
            nombre  = r["nombre"]
            val     = r["valoracion"]
            pct     = r.get("_pct_positivo", 0)
            dist    = r.get("distancia_km")
            platos  = r.get("platos_destacados", [])
            resumen = r.get("resumen", "")
            lineas.append(f"**{nombre}**")
            meta_linea = f"Valoración: {val}"
            if pct:
                meta_linea += f" · {pct:.0f}% reseñas positivas"
            if r.get("rango_precio"):
                from_mapa = {"euro": "€", "euro euro": "€€", "euro euro euro": "€€€", "euro euro euro euro": "€€€€"}
                meta_linea += f" · {from_mapa.get(r['rango_precio'], r['rango_precio'])}"
            lineas.append(meta_linea)
            if resumen:
                lineas.append(resumen[:120] + ("..." if len(resumen) > 120 else ""))
            if platos:
                lineas.append(f"🍽️ Platos: {', '.join(platos[:4])}")
            lineas.append("")
        return "\n".join(lineas).strip()

    ETIQUETAS_COCINA = {
        'taberna': 'tabernas', 'asador': 'asadores',
        'marisqueria': 'marisquerías', 'arroceria': 'arrocerías',
        'hamburgueseria': 'hamburgueserías', 'fusion': 'cocina fusión',
        'gallega': 'cocina gallega', 'vasca': 'cocina vasca',
        'asturiana': 'cocina asturiana', 'madrileña': 'cocina madrileña',
        'andaluza': 'cocina andaluza', 'italiana': 'cocina italiana',
        'japonesa': 'cocina japonesa', 'india': 'cocina india',
        'mexicana': 'cocina mexicana', 'peruana': 'cocina peruana',
        'venezolana': 'cocina venezolana', 'argentina': 'cocina argentina',
        'tailandesa': 'cocina tailandesa', 'arabe': 'cocina árabe',
        'francesa': 'cocina francesa', 'griega': 'cocina griega',
        'china': 'cocina china', 'colombiana': 'cocina colombiana',
        'americana': 'cocina americana',
    }
    if cocina:
        intro_parts.append(ETIQUETAS_COCINA.get(cocina, f'cocina {cocina}'))
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


# ── Log de consultas en memoria ──────────────────────────────────────────────
import threading as _threading
from datetime import datetime as _datetime

_historial_memoria: list = []   # se resetea al redesplegar
_log_lock = _threading.Lock()
ADMIN_KEY = os.environ.get("ADMIN_KEY", "tfm2024admin")  # configura en Render → Environment

def _guardar_log(consulta: str, respuesta: str, restaurantes: list):
    """Guarda cada consulta en memoria."""
    try:
        nombres = ", ".join(r.get("nombre", "") for r in restaurantes[:5])
        entrada = {
            "fecha":        _datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "consulta":     consulta,
            "respuesta":    respuesta.replace("\n", " ")[:600],
            "restaurantes": nombres,
            "n_resultados": len(restaurantes),
        }
        with _log_lock:
            _historial_memoria.append(entrada)
    except Exception as e:
        print(f"  [log] error: {e}")


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

        # Guardar en log
        _guardar_log(request.consulta, respuesta, restaurantes)

        return RecomendacionResponse(
            respuesta=respuesta,
            proyecto="nlptown/bert + análisis NLP local",
            restaurantes=restaurantes,
            consulta_usuario=request.consulta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/historial")
def admin_historial(key: str = ""):
    """Endpoint privado — muestra el historial de consultas en HTML."""
    if key != ADMIN_KEY:
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<h2>403 — Acceso denegado</h2>", status_code=403)
    from fastapi.responses import HTMLResponse
    with _log_lock:
        entradas = list(reversed(_historial_memoria))
    filas = ""
    for e in entradas:
        filas += f"""
        <tr>
          <td>{e['fecha']}</td>
          <td>{e['consulta']}</td>
          <td style='color:#888;font-size:12px'>{e['restaurantes']}</td>
          <td style='text-align:center'>{e['n_resultados']}</td>
          <td style='font-size:11px;color:#666'>{e['respuesta'][:200]}...</td>
        </tr>"""
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Historial de consultas</title>
  <style>
    body {{ font-family: sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }}
    h1 {{ color: #c8a96e; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{ background: #1a1a1a; color: #c8a96e; padding: 10px; text-align: left; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #2a2a2a; vertical-align: top; }}
    tr:hover td {{ background: #161616; }}
  </style>
</head>
<body>
  <h1>📋 Historial de consultas</h1>
  <p style='color:#666'>{len(entradas)} consultas desde el último despliegue</p>
  <table>
    <tr>
      <th>Fecha</th><th>Consulta</th><th>Restaurantes</th><th>N</th><th>Respuesta</th>
    </tr>
    {filas if filas else "<tr><td colspan='5' style='color:#555;text-align:center'>Sin consultas aún</td></tr>"}
  </table>
</body>
</html>"""
    return HTMLResponse(html)


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
