"""
regenerar_criterios.py
──────────────────────
Calcula criterios directamente desde las reseñas sin Gemini.
Rápido, sin coste, sin bloqueos.

Uso: python3.11 regenerar_criterios.py
"""
import os, unicodedata, pandas as pd
from collections import defaultdict

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_AN      = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
CSV_RESENAS = os.path.join(BASE_DIR, "resenas_unificadas.csv")

def norm(s):
    s = str(s).lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

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

# Mínimo de reseñas con señal positiva para marcar como True
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

def calcular_criterios(resenas_texto):
    resultado = {c: False for c in CRITERIOS}
    frases    = {c: [] for c in CRITERIOS}
    conteos   = defaultdict(int)

    for texto_original in resenas_texto:
        texto = norm(str(texto_original))
        for criterio, keywords in CRITERIOS.items():
            for kw in keywords:
                kw_n = norm(kw)
                if kw_n in texto and not tiene_negacion(texto, kw_n):
                    conteos[criterio] += 1
                    if len(frases[criterio]) < 2:
                        idx = texto.find(kw_n)
                        frag = texto[max(0,idx-50):idx+80].strip()
                        frases[criterio].append(frag)
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

# ── FIX: pre-inicializar columnas con dtype correcto antes del loop ────────────
# Las booleanas como False (bool), las de frases como "" (object/string).
# Sin esto, pandas infiere float64 en la primera asignación y luego explota
# al intentar meter texto.
for criterio in CRITERIOS:
    df[f"criterio_{criterio}"]        = False          # dtype bool
    df[f"criterio_{criterio}_frases"] = ""             # dtype object (string)
# ──────────────────────────────────────────────────────────────────────────────

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
