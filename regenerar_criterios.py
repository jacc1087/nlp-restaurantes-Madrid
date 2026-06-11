"""
regenerar_criterios.py
──────────────────────
Calcula criterios directamente desde las reseñas sin Gemini ni APIs externas.

Mejoras v4:
  - extraer_oracion() valida que la keyword esté en la oración devuelta
  - MIN_MENCIONES más estrictos para criterios ambiguos
  - Si la oración no es suficientemente relevante, se descarta
  - Sin APIs externas

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
    texto = texto.strip()
    if not texto:
        return texto
    texto = texto[0].upper() + texto[1:]
    if texto[-1] not in ".!?…":
        texto += "."
    return texto

NEGACIONES = {"no ","sin ","nunca ","tampoco ","ni ","jamas ","nada ","apenas "}

CRITERIOS = {
    "ninos":              ["niño","niña","bebe","bebé","infantil","sillita","trona","peques","pequeños","familia con niños","con niños","los niños","mis niños","nuestros niños"],
    "mascotas":           ["perro","mascota","admiten perros","dog friendly","pet friendly","peludo","con perro"],
    "terraza":            ["terraza","al aire libre","patio exterior","veladores","mesa fuera","en la terraza"],
    "vistas":             ["vistas","panoramica","panorámica","azotea","rooftop","mirador","vistas al"],
    "musica_directo":     ["musica en directo","música en directo","concierto","actuacion en vivo","en vivo","jazz en vivo","flamenco en vivo"],
    "romantico":          ["romantico","romántico","cena romantica","cena romántica","velas","para parejas","muy intimo","muy íntimo"],
    "buen_postre":        ["postre","postres","tarta","helado","tiramisu","tiramisú","mousse","brownie","coulant","de postre","los postres","el postre"],
    "precio_calidad":     ["calidad precio","calidad-precio","relacion calidad precio","precio razonable","precio asequible","precio justo","buena relacion calidad","merece la pena"],
    "grupos_grandes":     ["grupo grande","celebracion","celebración","cumpleaños","cumpleanos","cena de empresa","comida de empresa","reserva de grupo","para grupos"],
    "vegano_vegetariano": ["vegano","vegana","vegetariano","vegetariana","opciones veganas","sin carne","plant based","menú vegano","carta vegana"],
    "sin_gluten":         ["sin gluten","celiaco","celiaca","celíaco","celíaca","gluten free","intolerancia al gluten","apto celiaco"],
}

# Más estrictos: si hay dudas, que no salga
MIN_MENCIONES = {
    "ninos":              3,
    "mascotas":           2,
    "terraza":            3,
    "vistas":             2,
    "musica_directo":     2,
    "romantico":          3,
    "buen_postre":        4,   # ← más exigente, era 3
    "precio_calidad":     4,   # ← más exigente, era 3
    "grupos_grandes":     3,
    "vegano_vegetariano": 2,
    "sin_gluten":         2,
}

def tiene_negacion(texto, keyword, ventana=25):
    idx = texto.find(keyword)
    if idx == -1: return False
    contexto = texto[max(0, idx-ventana):idx]
    return any(neg in contexto for neg in NEGACIONES)

def extraer_oracion(texto_original: str, keyword_norm: str, max_chars: int = 180) -> str:
    """
    Extrae la oración del texto ORIGINAL que contiene la keyword.
    Valida que la keyword realmente esté en el fragmento devuelto.
    Si no puede garantizarlo, devuelve "".
    """
    texto_n = norm(texto_original)
    idx = texto_n.find(keyword_norm)
    if idx == -1:
        return ""

    # Buscar inicio de oración mirando hacia atrás
    inicio = max(0, idx - 300)
    segmento_antes = texto_original[inicio:idx]
    m = list(re.finditer(r'[.!?\n]\s*', segmento_antes))
    inicio_oracion = inicio + m[-1].end() if m else inicio

    # Buscar fin de oración mirando hacia delante
    segmento_despues = texto_original[idx:]
    m2 = re.search(r'[.!?\n]', segmento_despues)
    fin_oracion = idx + m2.end() if m2 else min(idx + 250, len(texto_original))

    oracion = texto_original[inicio_oracion:fin_oracion].strip()

    # ── VALIDACIÓN CLAVE: la keyword tiene que estar en la oración extraída ──
    if keyword_norm not in norm(oracion):
        # Fallback: tomar ventana directa centrada en la keyword
        oracion = texto_original[max(0, idx-80):min(len(texto_original), idx+120)].strip()
        if keyword_norm not in norm(oracion):
            return ""  # No se puede garantizar relevancia → descartar

    # Truncar sin cortar palabra
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

    # Si un criterio es True pero no tiene frases válidas → quitar el criterio
    for criterio in CRITERIOS:
        if resultado[criterio] and not frases[criterio]:
            resultado[criterio] = False

    return resultado, {c: " | ".join(v) for c, v in frases.items() if v and resultado[c]}

# ── Main ──────────────────────────────────────────────────────────────────────
print("Cargando datos...")
df      = pd.read_csv(CSV_AN)
resenas = pd.read_csv(CSV_RESENAS)

col_id  = next(c for c in resenas.columns if c.lower() == "id_restaurante")
col_rev = next(c for c in resenas.columns if c.lower() in ("review","texto","resena"))
resenas[col_id] = resenas[col_id].astype(int).astype(str)
df["id_restaurante"] = df["id_restaurante"].astype(int).astype(str)

# Pre-inicializar columnas con dtype correcto
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
