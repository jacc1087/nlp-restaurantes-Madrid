"""
Análisis de reseñas por restaurante
Sentimiento con nlptown/bert-base-multilingual-uncased-sentiment

Uso:
    python3.11 analizar_restaurante.py

Requisitos:
    pip install transformers torch nltk scikit-learn pandas
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
RESTAURANTE_ID = 2
CSV_PATH       = "resenas_unificadas.csv"
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELO_HF      = "nlptown/bert-base-multilingual-uncased-sentiment"
MIN_MENCIONES  = 10   # mínimo para considerar una dimensión representativa
MIN_CONFIANZA  = 0.50 # confianza mínima del modelo; por debajo se marca como dudosa

STOP_ES = set(stopwords.words('spanish')) | {
    'si', 'así', 'tan', 'ser', 'año', 'vez', 'bien', 'muy', 'más', 'menos',
    'todo', 'toda', 'todos', 'cada', 'este', 'esta', 'estos', 'estas',
    'aún', 'ahí', 'allí', 'aquí', 'hay', 'era', 'fue', 'han', 'has',
    'hemos', 'nos', 'les', 'lo', 'la', 'las', 'los', 'del', 'al',
    'un', 'una', 'unos', 'unas', 'su', 'sus', 'mi', 'mis', 'se', 'me',
    'te', 'le', 'que', 'de', 'en', 'y', 'a', 'el', 'es', 'por', 'con',
    'para', 'como', 'no', 'pero', 'o', 'si', 'cuando', 'donde', 'aunque',
}

# (nombre_a_mostrar, [variantes_a_buscar])
PLATOS_GRUPOS = [
    ('tikka masala',        ['tikka masala', 'tikka']),
    ('pollo (genérico)',    ['pollo']),
    ('queso paneer',        ['queso', 'paneer']),
    ('naan / pan',          ['naan', 'nan']),
    ('mango lassi / mango', ['mango lassi', 'mango']),
    ('biryani',             ['biryani']),
    ('masala (genérico)',   ['masala']),
    ('arroz / basmati',     ['arroz', 'basmati']),
    ('samosa',              ['samosa']),
    ('cordero / lamb',      ['cordero', 'lamb']),
    ('curry',               ['curry']),
    ('korma',               ['korma']),
    ('verduras / vegetal',  ['verdura', 'vegetal', 'vegetariano', 'vegano']),
    ('butter chicken',      ['butter chicken']),
    ('vindaloo',            ['vindaloo']),
]

CAMAREROS = []   # se detectan automáticamente abajo

DIMENSIONES_KEYWORDS = {
    'servicio':  ['servicio', 'camarero', 'camarera', 'atención', 'atento', 'amable',
                  'profesional', 'trato', 'pendiente', 'recomendó', 'recomendaron'],
    'comida':    ['comida', 'plato', 'carta', 'sabor', 'calidad', 'producto', 'fresco',
                  'rico', 'riquísimo', 'delicioso', 'exquisito', 'auténtico', 'especias'],
    'ambiente':  ['ambiente', 'local', 'decoración', 'acogedor', 'bonito',
                  'atmósfera', 'espacio', 'tranquilo', 'íntimo', 'cálido'],
    'precio':    ['precio', 'caro', 'barato', 'relación', 'calidad-precio', 'coste', 'euros'],
    'velocidad': ['rápido', 'lento', 'espera', 'tardó', 'tiempo', 'prisas', 'ágil'],
    'ruido':     ['ruido', 'ruidoso', 'silencioso', 'tranquilo', 'música'],
    'limpieza':  ['limpio', 'limpieza', 'higiene', 'aseado'],
}

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────
def limpiar(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def tokenizar(texto: str) -> list:
    tokens = word_tokenize(limpiar(texto), language='spanish')
    return [t for t in tokens if t not in STOP_ES and len(t) > 2]

def estrellas_a_categoria(e: int) -> str:
    if e >= 4: return 'positivo'
    if e == 3: return 'neutro'
    return 'negativo'

def analizar_sentimiento(texto: str) -> tuple:
    r = sentiment_pipeline(str(texto))[0]
    e = int(r['label'].split()[0])
    return e, round(r['score'], 3), estrellas_a_categoria(e)

def porcentaje_menciones(serie, keywords):
    total = len(serie)
    n = sum(1 for r in serie if any(kw in limpiar(str(r)) for kw in keywords))
    return n, round(n / total * 100, 1)

# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────
print("=" * 60)
print("Cargando modelo nlptown (primera vez descarga ~700 MB)...")
print("=" * 60)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=MODELO_HF,
    truncation=True, max_length=512,
)
print("Modelo listo.\n")

# ─────────────────────────────────────────────
# CARGA Y ANÁLISIS
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
r1 = df[df['Id_Restaurante'] == RESTAURANTE_ID].copy()
nombre     = r1['Restaurante'].iloc[0]
valoracion = r1['Valoracion'].mean()
n_reviews  = len(r1)

print(f"Analizando {n_reviews} reseñas de «{nombre}»...")
print("Calculando sentimiento (puede tardar 1-2 min)...\n")

resultados         = r1['Review'].apply(lambda x: analizar_sentimiento(str(x)))
r1['estrellas']    = resultados.apply(lambda x: x[0])
r1['score_modelo'] = resultados.apply(lambda x: x[1])
r1['sentimiento']  = resultados.apply(lambda x: x[2])
r1['tokens']       = r1['Review'].apply(lambda x: tokenizar(str(x)))
r1['texto_limpio'] = r1['Review'].apply(limpiar)
r1['baja_confianza'] = r1['score_modelo'] < MIN_CONFIANZA

sent_global      = Counter(r1['sentimiento'])
estrellas_dist   = Counter(r1['estrellas'])
n_baja_confianza = r1['baja_confianza'].sum()
avg_stars        = round(r1['estrellas'].mean(), 2)
print("Sentimiento calculado.\n")

# ─────────────────────────────────────────────
# DIMENSIONES
# ─────────────────────────────────────────────
dim_stats = {}
for dim, kws in DIMENSIONES_KEYWORDS.items():
    menciones, pct = porcentaje_menciones(r1['Review'], kws)
    subset = r1[r1['Review'].apply(lambda x: any(kw in limpiar(str(x)) for kw in kws))]
    sc = Counter(subset['sentimiento'])
    dim_stats[dim] = {
        'menciones':      menciones,
        'pct':            pct,
        'positivo':       sc.get('positivo', 0),
        'negativo':       sc.get('negativo', 0),
        'neutro':         sc.get('neutro', 0),
        'avg_estrellas':  round(subset['estrellas'].mean(), 2) if len(subset) else 0,
        'representativa': menciones >= MIN_MENCIONES,
    }

# ─────────────────────────────────────────────
# PLATOS
# ─────────────────────────────────────────────
platos_freq = Counter()
for nombre_plato, variantes in PLATOS_GRUPOS:
    cnt = sum(1 for r in r1['Review'] if any(v in str(r).lower() for v in variantes))
    if cnt > 0:
        platos_freq[nombre_plato] = cnt
platos_top = platos_freq.most_common(15)

# ─────────────────────────────────────────────
# PERSONAL (detección automática por nombres propios frecuentes)
# ─────────────────────────────────────────────
nombres_candidatos = Counter()
for r in r1['Review']:
    for w in re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{2,}\b', str(r)):
        nombres_candidatos[w.lower()] += 1
# Filtrar stopwords y palabras comunes
excluir = STOP_ES | {'gracias', 'madrid', 'lugar', 'restaurante', 'comida',
                      'servicio', 'ambiente', 'experiencia', 'buena', 'muy'}
cam_freq = Counter({k: v for k, v in nombres_candidatos.items()
                    if k not in excluir and v >= 2})

# ─────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────
vectorizer  = TfidfVectorizer(stop_words=list(STOP_ES), min_df=2,
                               max_features=100, ngram_range=(1, 2))
X           = vectorizer.fit_transform(r1['texto_limpio'])
mean_tfidf  = np.asarray(X.mean(axis=0)).flatten()
top_idx     = mean_tfidf.argsort()[::-1][:20]
tfidf_top   = [(vectorizer.get_feature_names_out()[i], round(mean_tfidf[i], 4))
               for i in top_idx]

# ─────────────────────────────────────────────
# RESEÑAS NEGATIVAS Y DUDOSAS
# ─────────────────────────────────────────────
negativas = r1[r1['sentimiento'] == 'negativo'].sort_values('estrellas')
dudosas   = r1[r1['baja_confianza'] & (r1['sentimiento'] != 'negativo')].sort_values('score_modelo')

# ─────────────────────────────────────────────
# INFORME TXT
# ─────────────────────────────────────────────
lineas = []
sep  = "=" * 65
sep2 = "-" * 65

def h1(t): lineas.append(f"\n{sep}\n  {t.upper()}\n{sep}")
def h2(t): lineas.append(f"\n{sep2}\n  {t}\n{sep2}")
def li(t): lineas.append(f"  • {t}")
def br():  lineas.append("")

h1(f"informe de análisis — {nombre}")
lineas.append(f"  Restaurante ID  : {RESTAURANTE_ID}")
lineas.append(f"  Nombre          : {nombre}")
lineas.append(f"  Nº de reseñas   : {n_reviews}")
lineas.append(f"  Valoración media: {valoracion:.2f} / 5.0")
lineas.append(f"  Modelo          : {MODELO_HF}")
br()

h2("1. Distribución de estrellas (modelo)")
for e in range(5, 0, -1):
    cnt = estrellas_dist.get(e, 0)
    pct = round(cnt / n_reviews * 100, 1)
    lineas.append(f"  {'★'*e:5s}  {cnt:3d} reseñas ({pct:5.1f}%)  {'█' * min(cnt, 40)}")
br()
lineas.append(f"  Media de estrellas según el modelo: {avg_stars} / 5.0")
br()

h2("2. Sentimiento global (agrupado)")
total_sent = sum(sent_global.values())
for s, emoji in [('positivo', '✅'), ('neutro', '➖'), ('negativo', '❌')]:
    cnt = sent_global.get(s, 0)
    pct = round(cnt / total_sent * 100, 1)
    lineas.append(f"  {emoji}  {s.capitalize():12s}: {cnt:3d} reseñas  ({pct}%)")
br()

h2("3. Análisis por dimensión")
lineas.append(f"  ⚠️  Dimensiones con menos de {MIN_MENCIONES} menciones = NO REPRESENTATIVAS.")
for dim, st in dim_stats.items():
    etiqueta = "" if st['representativa'] else "  ⚠️  NO REPRESENTATIVA"
    lineas.append(f"\n  [{dim.upper()}]  — {st['menciones']}/{n_reviews} reseñas ({st['pct']}%){etiqueta}")
    lineas.append(f"    ✅ Positivo: {st['positivo']}   ❌ Negativo: {st['negativo']}   ➖ Neutro: {st['neutro']}")
    fiable = "" if st['representativa'] else "  (no fiable, muestra insuficiente)"
    lineas.append(f"    Media estrellas: {st['avg_estrellas']} ★{fiable}")
br()

h2("4. Platos más mencionados")
for plato, cnt in platos_top:
    lineas.append(f"  {'█' * cnt}  {plato} ({cnt}x)")
br()

h2("5. Personal mencionado")
if cam_freq:
    for cam, cnt in cam_freq.most_common(10):
        lineas.append(f"  {cam.capitalize():<18s} {cnt}x")
else:
    lineas.append("  No se detectaron nombres de personal con frecuencia suficiente.")
br()

h2("6. Términos más característicos (TF-IDF)")
for term, score in tfidf_top:
    lineas.append(f"  {term:<30s} {score}")
br()

h2("7. Reseñas con valoración negativa")
if len(negativas):
    for _, row in negativas.iterrows():
        flag = "  ⚠️  CONFIANZA BAJA — revisar" if row['baja_confianza'] else ""
        lineas.append(f"  [{'★' * row['estrellas']}  confianza: {row['score_modelo']}]{flag}")
        lineas.append(f"  → \"{row['Review'][:220]}\"")
        br()
else:
    lineas.append("  No se detectaron reseñas negativas.")
br()

h2("7b. Reseñas con baja confianza (posibles errores del modelo)")
lineas.append(f"  Umbral: {MIN_CONFIANZA} — se recomienda revisión manual.")
br()
if len(dudosas):
    for _, row in dudosas.iterrows():
        lineas.append(f"  [{'★' * row['estrellas']} {row['sentimiento']:8s}  confianza: {row['score_modelo']}]")
        lineas.append(f"  → \"{row['Review'][:220]}\"")
        br()
else:
    lineas.append("  No se detectaron reseñas con baja confianza.")
br()

h2("8. Resumen ejecutivo")
pos_pct     = round(sent_global.get('positivo', 0) / total_sent * 100, 1)
neg_cnt     = sent_global.get('negativo', 0)
dims_no_rep = [d for d, st in dim_stats.items() if not st['representativa']]
dims_rep    = [d for d, st in dim_stats.items() if st['representativa']]
li(f"Valoración media (Google): {valoracion:.2f}/5 sobre {n_reviews} reseñas.")
li(f"Media de estrellas según el modelo: {avg_stars}/5.")
li(f"{pos_pct}% de las reseñas clasificadas como positivas (4-5 estrellas).")
if platos_top:
    top3 = ', '.join([p for p, _ in platos_top[:3]])
    li(f"Platos más mencionados: {top3}.")
if cam_freq:
    top_cam = cam_freq.most_common(1)[0]
    li(f"Personal destacado: '{top_cam[0].capitalize()}' ({top_cam[1]}x).")
if dims_rep:
    dim_mejor = max(dims_rep, key=lambda d: dim_stats[d]['avg_estrellas'])
    li(f"Dimensión mejor valorada: '{dim_mejor}' ({dim_stats[dim_mejor]['avg_estrellas']}★).")
li(f"Reseñas negativas: {neg_cnt}.")
li(f"Reseñas con baja confianza: {n_baja_confianza} — revisar manualmente.")
if dims_no_rep:
    li(f"Dimensiones NO representativas: {', '.join(dims_no_rep)}.")
br()

ruta_txt = os.path.join(OUTPUT_DIR, f"informe_restaurante_{RESTAURANTE_ID}.txt")
with open(ruta_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(lineas))

print(f"✔ Informe guardado: {ruta_txt}")
