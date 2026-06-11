"""
regenerar_criterios.py
──────────────────────
Calcula criterios directamente desde las reseñas sin Gemini ni APIs externas.

Mejoras v5:
  - partir_en_segmentos: corta por puntuación, comas Y conectores "y + artículo"
  - extraer_oracion: nunca trunca — si no cabe completo, descarta
  - limpiar_frase: elimina conectores sueltos al inicio del fragmento

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
    # eliminar conectores sueltos al inicio (artefacto del split por "y")
    texto = re.sub(r'^(y |e |pero |aunque |también |además )', '', texto, flags=re.IGNORECASE).strip()
    if not texto:
        return texto
    texto = texto[0].upper() + texto[1:]
    if texto[-1] not in ".!?":
        texto += "."
    return texto

NEGACIONES = {"no ","sin ","nunca ","tampoco ","ni ","jamas ","nada ","apenas "}

CRITERIOS = {
    "ninos":              ["niño","niña","bebe","bebé","infantil","sillita","trona","peques","pequeños","familia con niños","con niños","los niños","mis niños","nuestros niños"],
    "mascotas":           ["perro","mascota","admiten perros","dog friendly","pet friendly","peludo","con perro"],
    "terraza":            ["terraza","al aire libre","patio exterior","veladores","mesa fuera","en la terraza"],
    "vistas":             ["vistas al mar","vistas al rio","vistas a la ciudad","vistas panoramicas","vistas panorámicas","panoramica","panorámica","azotea con vistas","rooftop","mirador","con vistas"],
    "musica_directo":     ["musica en directo","música en directo","concierto","actuacion en vivo","en vivo","jazz en vivo","flamenco en vivo"],
    "romantico":          ["romantico","romántico","cena romantica","cena romántica","velas","para parejas","muy intimo","muy íntimo"],
    "buen_postre":        ["postre","postres","tarta","helado","tiramisu","tiramisú","mousse","brownie","coulant","de postre","los postres","el postre"],
    "precio_calidad":     ["calidad precio","calidad-precio","relacion calidad precio","precio razonable","precio asequible","precio justo","buena relacion calidad","merece la pena"],
    "grupos_grandes":     ["grupo grande","celebracion","celebración","cumpleaños","cumpleanos","cena de empresa","comida de empresa","reserva de grupo","para grupos"],
    "vegano_vegetariano": ["vegano","vegana","vegetariano","vegetariana","opciones veganas","sin carne","plant based","menú vegano","carta vegana"],
    "sin_gluten":         ["sin gluten","celiaco","celiaca","celíaco","celíaca","gluten free","intolerancia al gluten","apto celiaco"],
}

MIN_MENCIONES = {
    "ninos":              3,
    "mascotas":           2,
    "terraza":            3,
    "vistas":             3,
    "musica_directo":     2,
    "romantico":          3,
    "buen_postre":        4,
    "precio_calidad":     4,
    "grupos_grandes":     3,
    "vegano_vegetariano": 2,
    "sin_gluten":         2,
}

def tiene_negacion(texto, keyword, ventana=25):
    idx = texto.find(keyword)
    if idx == -1: return False
    contexto = texto[max(0, idx-ventana):idx]
    return any(neg in contexto for neg in NEGACIONES)

def partir_en_segmentos(texto):
    """
    Divide el texto en segmentos usando como delimitadores:
      1. Puntuación fuerte: . ! ? newline
      2. Cualquier coma
      3. " y " seguido de artículo o adverbio frecuente
    Devuelve lista de (inicio, fin) sobre el texto original.
    """
    patron = re.compile(
        r'[.!?\n]+'
        r'|,\s*(?=\w)'
        r'| y (?=(?:el |la |los |las |un |una |todo |muy |lo |también |además |super |súper |fue |es |era |están |estaba ))',
        re.IGNORECASE
    )
    cortes = [0]
    for m in patron.finditer(texto):
        cortes.append(m.start())
        cortes.append(m.end())
    cortes.append(len(texto))
    cortes = sorted(set(cortes))
    segs = []
    for i in range(len(cortes) - 1):
        ini, fin = cortes[i], cortes[i+1]
        if len(texto[ini:fin].strip()) > 8:
            segs.append((ini, fin))
    return segs

def extraer_oracion(texto_original, keyword_norm, max_chars=120):
    """
    Extrae el segmento COMPLETO que contiene la keyword.
    Regla de oro: si no cabe completo (<= max_chars), DESCARTA.
    Nunca trunca. Nunca muestra texto incompleto.
    """
    texto_n = norm(texto_original)
    idx_n   = texto_n.find(keyword_norm)
    if idx_n == -1:
        return ""

    for ini, fin in partir_en_segmentos(texto_original):
        if ini <= idx_n < fin:
            fragmento = texto_original[ini:fin].strip()
            if keyword_norm not in norm(fragmento):
                continue
            if len(fragmento) > max_chars:
                return ""
            return limpiar_frase(fragmento)

    return ""

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

    # criterio True pero sin frases válidas → desactivar
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
