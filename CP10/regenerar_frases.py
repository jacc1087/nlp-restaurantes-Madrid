"""
regenerar_frases.py
───────────────────
Regenera SOLO las columnas servicio_frases y resenas_destacadas
del analisis_restaurantes.csv usando Gemini con tokens suficientes.

NO toca ninguna otra columna ni reprocesa criterios.
Tarda ~10 minutos (180 restaurantes × 2 columnas × pausa 0.5s).

Uso:
    python3.11 regenerar_frases.py
"""

import os, json, time, pandas as pd, urllib.request as _ureq

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_AN      = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
CSV_RESENAS = os.path.join(BASE_DIR, "resenas_unificadas.csv")

GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Cargar .env si existe ─────────────────────────────────────────────────────
_env = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env) and not GEMINI_KEY:
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    GEMINI_KEY = v.strip()

if not GEMINI_KEY:
    print("ERROR: GEMINI_API_KEY no encontrada. Añádela al .env o como variable de entorno.")
    exit(1)

# ── Gemini ────────────────────────────────────────────────────────────────────
def gemini(prompt: str, max_tokens: int = 300) -> str:
    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens},
    }).encode()
    url = (f"https://generativelanguage.googleapis.com/v1/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
    for intento in range(3):
        try:
            req  = _ureq.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
            resp = _ureq.urlopen(req, timeout=25)
            result = json.loads(resp.read())
            texto = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            time.sleep(0.4)
            return texto
        except Exception as e:
            if any(c in str(e) for c in ["429","503","500"]) and intento < 2:
                time.sleep(4)
            else:
                return ""
    return ""

def es_cortada(texto: str) -> bool:
    """True si la frase parece cortada o inútil."""
    t = texto.strip()
    return not t or t.lower() in ("nan","none","") or len(t) < 25

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(CSV_AN)
resenas = pd.read_csv(CSV_RESENAS)

col_id  = next(c for c in resenas.columns if "id_restaurante" in c.lower() or c == "Id_Restaurante")
col_rev = next(c for c in resenas.columns if c.lower() in ("review","texto","resena"))
col_val = next((c for c in resenas.columns if c.lower() in ("estrellas","valoracion")), None)

resenas[col_id] = resenas[col_id].astype(str)
df["id_restaurante"] = df["id_restaurante"].astype(str)

print(f"  {len(df)} restaurantes | {len(resenas)} reseñas\n")

# ── Regenerar por restaurante ─────────────────────────────────────────────────
nuevas_servicio  = []
nuevas_destacadas = []

for i, row in df.iterrows():
    rid    = str(row["id_restaurante"])
    nombre = str(row["nombre"])
    print(f"[{i+1}/{len(df)}] {nombre[:40]}...", end=" ", flush=True)

    df_r = resenas[resenas[col_id] == rid]

    # ── servicio_frases ───────────────────────────────────────────────────────
    serv_actual = str(row.get("servicio_frases", "") or "")
    if es_cortada(serv_actual):
        kws = ["atención","servicio","amable","trato","camarero","camarera","personal"]
        frags = []
        for _, rv in df_r.iterrows():
            texto = str(rv[col_rev]).lower()
            if any(k in texto for k in kws):
                frags.append(str(rv[col_rev])[:180])
            if len(frags) >= 3:
                break

        if frags:
            frags_txt = " | ".join(frags[:3])
            prompt = (
                f"Reescribe en 2 frases naturales y completas lo que expresan "
                f"estas opiniones sobre el servicio de {nombre}. "
                f"Sin comillas, sin nombres propios, sin reproducir texto literal. "
                f"Separa las frases con ' | '.\n"
                f"Opiniones: {frags_txt}\nFrases:"
            )
            serv_nuevo = gemini(prompt, max_tokens=250)
            nuevas_servicio.append(serv_nuevo if not es_cortada(serv_nuevo) else serv_actual)
        else:
            nuevas_servicio.append(serv_actual)
    else:
        nuevas_servicio.append(serv_actual)

    # ── resenas_destacadas ────────────────────────────────────────────────────
    dest_actual = str(row.get("resenas_destacadas", "") or "")
    if es_cortada(dest_actual):
        if col_val:
            df_pos = df_r[pd.to_numeric(df_r[col_val], errors="coerce") >= 4].copy()
        else:
            df_pos = df_r.copy()
        df_pos = df_pos.copy()
        df_pos["_len"] = df_pos[col_rev].fillna("").astype(str).apply(len)
        df_pos = df_pos.sort_values("_len", ascending=False).head(4)

        if len(df_pos) > 0:
            frags = df_pos[col_rev].astype(str).tolist()[:3]
            frags_txt = " | ".join(f[:200] for f in frags)
            prompt = (
                f"Resume en 3 frases cortas, naturales y completas "
                f"las mejores opiniones de clientes sobre {nombre}. "
                f"Cada frase separada por ' | '. "
                f"Sin comillas, sin nombres propios, sin reproducir texto literal.\n"
                f"Opiniones: {frags_txt}\nResumen:"
            )
            dest_nuevo = gemini(prompt, max_tokens=350)
            nuevas_destacadas.append(dest_nuevo if not es_cortada(dest_nuevo) else "")
        else:
            nuevas_destacadas.append("")
    else:
        nuevas_destacadas.append(dest_actual)

    print("✔")

# ── Guardar ───────────────────────────────────────────────────────────────────
df["servicio_frases"]   = nuevas_servicio
df["resenas_destacadas"] = nuevas_destacadas

df.to_csv(CSV_AN, index=False)

cortadas_serv = sum(1 for v in nuevas_servicio  if es_cortada(v))
cortadas_dest = sum(1 for v in nuevas_destacadas if es_cortada(v))
print(f"\n✔ CSV actualizado.")
print(f"  servicio_frases:   {len(df)-cortadas_serv}/{len(df)} correctas")
print(f"  resenas_destacadas: {len(df)-cortadas_dest}/{len(df)} correctas")
