"""
regenerar_criterios.py
──────────────────────
Calcula criterios directamente desde las reseñas sin Gemini ni APIs externas.
Rápido, sin coste, sin bloqueos.

Mejoras v3:
  - Fragmentos extraídos del texto ORIGINAL (no del normalizado)
  - Fragmento centrado en la oración que contiene la keyword
  - limpiar_frase(): capitaliza y añade punto final (sin API)

Uso: python3.11 regenerar_criterios.py
"""
import os, re, unicodedata, pandas as pd
from collections import defaultdict

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_AN      = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
CSV_RESENAS = os.path.join(BASE_DIR, "resenas_unificadas.csv")

def norm(s):
    s = str(s).lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def limpiar_frase(texto: str) -> str:
    """Capitaliza primera letra y añade punto final si falta. Sin API."""
    texto = texto.strip()
    if not texto:
        return texto
    texto = texto[0].upper() + texto[1:]
    if texto[-1] not in ".!?…":
        texto += "."
    return texto

NEGACIONES = {"no ","sin ","nunca ","tampoco ","ni ","jamas ","nada ","apenas "}

CRITERIOS = {
    "ninos":              ["niño","niña","bebe","bebé","infantil","sillita","trona","peques","pequeños","familia con niños"],
    "mascotas":           ["perro","mascota","admiten perros","dog friendly","pet friendly","peludo"],
    "terraza":            ["terraza","al aire libre","patio exterior","veladores","mesa fuera"],
    "vistas":             ["vistas","panoramica","panorámica","azotea","rooftop","mirador"],
    "musica_directo":     ["musica en directo","música en directo","concierto","actuacion","en vivo","jazz","flamenco en vivo"],
    "romantico":          ["romantico","romántico","intimo","íntimo","cena romantica","velas","para parejas"],
    "buen_postre":        ["postre","postres","tarta","helado","tiramisu","tiramisú","mousse","brownie","coulant"],
    "precio_calidad":     ["calidad precio","calidad-precio","relacion calidad","precio razonable","precio asequible","precio justo","buena relacion calidad"],
    "grupos_grandes":     ["grupo grande","celebracion","celebración","cumpleaños","cumpleanos","evento","cena de empresa","comida de empresa","reserva de grupo"],
    "vegano_vegetariano": ["vegano","vegana","vegetariano","vegetariana","opciones veganas","sin carne","plant based","menú vegano"],
    "sin_gluten":         ["sin gluten","celiaco","celiaca","celíaco","celíaca","gluten free","intolerancia al gluten"],
}

MIN_MENCIONES = {
    "ninos": 2, "mascotas": 1, "terraza": 2, "vistas": 1,
    "musica_directo": 1, "romantico": 2, "buen_postre": 3,
    "precio_calidad": 3, "grupos_grandes": 2,
    "vegano_vegetariano": 1, "sin_gluten": 1,
}

def tiene_negacion(texto, keyword, ventana=25):
    idx = texto.find(keyword)
    if idx == -1: return False
    contexto = texto[max(0, idx-ventana):idx]
    return any(neg in contexto for neg in NEGACIONES)

def extraer_oracion(texto_original: str, keyword_norm: str, max_chars: int = 160) -> str:
    """
    Busca la keyword (normalizada) en el texto normalizado,
    pero devuelve la oración completa del texto ORIGINAL.
    """
    texto_n = norm(texto_original)
    idx = texto_n.find(keyword_norm)
    if idx == -1:
        return ""

    # Buscar inicio de oración
    inicio = max(0, idx - 200)
    segmento_antes = texto_original[inicio:idx]
    m = list(re.finditer(r'[.!?]\s+', segmento_antes))
    inicio_oracion = inicio + m[-1].end() if m else inicio

    # Buscar fin de oración
    segmento_despues = texto_original[idx:]
    m2 = re.search(r'[.!?]', segmento_despues)
    fin_oracion = idx + m2.end() if m2 else min(idx + 200, len(texto_original))

    oracion = texto_original[inicio_oracion:fin_oracion].strip()

    # Truncar sin cortar a mitad de palabra
    if len(oracion) > max_chars:
        oracion = oracion[:max_chars].rsplit(' ', 1)[0] + "…"

    return limpiar_frase(oracion)

def calcular_criterios(resenas_texto):
    resultado = {c: False for c in CRITERIOS}
    frases    = {c: [] for c in CRITERIOS}
    conteos   = defaultdict(int)

    for texto_original in resenas_texto:
        texto_n = norm(str(texto_original))
        for criterio, keywords in CRITERIOS.items():
            for kw in keywords:
                kw_n = norm(kw)
                if kw_n in texto_n and not tiene_negacion(texto_n, kw_n):
                    conteos[criterio] += 1
                    if len(frases[criterio]) < 2:
                        oracion = extraer_oracion(str(texto_original), kw_n)
                        if oracion:
                            frases[criterio].append(oracion)
                    break

    for criterio in CRITERIOS:
        if conteos[criterio] >= MIN_MENCIONES.get(criterio, 2):
            resultado[criterio] = True

    return resultado, {c: " | ".join(v) for c, v in frases.items() if v and resultado[c]}

# ── Main ──────────────────────────────────────────────────────────────────────
print("Cargando datos...")
df      = pd.read_csv(CSV_AN)
resenas = pd.read_csv(CSV_RESENAS)

col_id  = next(c for c in resenas.columns if c.lower() == "id_restaurante")
col_rev = next(c for c in resenas.columns if c.lower() in ("review","texto","resena"))
resenas[col_id] = resenas[col_id].astype(int).astype(str)
df["id_restaurante"] = df["id_restaurante"].astype(int).astype(str)

# Pre-inicializar columnas con dtype correcto (evita TypeError float64 → string)
for criterio in CRITERIOS:
    df[f"criterio_{criterio}"]        = False
    df[f"criterio_{criterio}_frases"] = ""

print(f"  {len(df)} restaurantes | {len(resenas)} reseñas\n")

for i, row in df.iterrows():
    rid    = str(row["id_restaurante"])
    nombre = str(row["nombre"])
    textos = resenas[resenas[col_id] == rid][col_rev].dropna().astype(str).tolist()

    resultado, frases = calcular_criterios(textos)

    for criterio, valor in resultado.items():
        df.at[i, f"criterio_{criterio}"] = valor
    for criterio, frase in frases.items():
        df.at[i, f"criterio_{criterio}_frases"] = frase

    activos = [c for c, v in resultado.items() if v]
    print(f"[{i+1}/180] {nombre[:40]:40} [{', '.join(activos) if activos else 'ninguno'}]")

    if (i + 1) % 20 == 0:
        df.to_csv(CSV_AN, index=False)
        print(f"  → CSV guardado [{i+1}/180]")

df.to_csv(CSV_AN, index=False)

print("\n=== RESUMEN ===")
for criterio in CRITERIOS:
    col = f"criterio_{criterio}"
    n = df[col].astype(str).str.lower().eq("true").sum()
    print(f"  {col}: {n}/180")

print("\n✔ Listo. Sube el analisis_restaurantes.csv a Render.")
