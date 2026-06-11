"""
generar_agente.py — genera agente_restaurantes.html con filtros de criterios
Uso: python3 generar_agente.py

Las coordenadas de cada restaurante se obtienen geocodificando la dirección
completa con Nominatim (OSM) la primera vez. Los resultados se guardan en
.cache_coords.json junto al script para no repetir llamadas en ejecuciones
posteriores. Solo se geocodifican restaurantes nuevos o sin coordenadas.
"""
import pandas as pd, json, os, math, time, urllib.request

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
OUTPUT_HTML = os.path.join(BASE_DIR, "agente_restaurantes.html")
CACHE_COORDS= os.path.join(BASE_DIR, ".cache_coords.json")

SOL_COORDS = [40.4168, -3.7038]  # Puerta del Sol — referencia "céntrico"

def haversine_py(a, b):
    R = 6371
    dLat = math.radians(b[0] - a[0])
    dLon = math.radians(b[1] - a[1])
    f = math.sin(dLat/2)**2 + math.cos(math.radians(a[0]))*math.cos(math.radians(b[0]))*math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(f), math.sqrt(1-f))

def _cargar_cache_coords():
    if os.path.exists(CACHE_COORDS):
        try:
            with open(CACHE_COORDS, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _guardar_cache_coords(cache):
    try:
        with open(CACHE_COORDS, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  No se pudo guardar caché de coords: {e}")

def _geocodificar_nominatim(direccion):
    """Geocodifica una dirección completa con Nominatim. Devuelve [lat, lon] o None."""
    query = f"{direccion}, Madrid, España"
    url = ('https://nominatim.openstreetmap.org/search?q='
           + urllib.request.quote(query)
           + '&format=json&limit=1&countrycodes=es')
    try:
        req = urllib.request.Request(url, headers={
            'Accept-Language': 'es',
            'User-Agent': 'agente-restaurantes-madrid-tfm'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data:
            return [round(float(data[0]['lat']), 6),
                    round(float(data[0]['lon']), 6)]
    except Exception as e:
        print(f"    ⚠️  Nominatim error para '{direccion}': {e}")
    return None

def geocodificar_restaurantes(records):
    """
    Para cada restaurante sin coordenadas en caché, llama a Nominatim.
    Guarda resultados en .cache_coords.json.
    Respeta el rate limit de Nominatim: 1 petición/segundo.
    """
    cache = _cargar_cache_coords()
    pendientes = [r for r in records
                  if str(r['id_restaurante']) not in cache
                  and r.get('direccion','').strip()]

    if pendientes:
        print(f"Geocodificando {len(pendientes)} restaurantes con Nominatim...")
        nuevos = 0
        for r in pendientes:
            rid  = str(r['id_restaurante'])
            addr = r['direccion'].strip()
            coords = _geocodificar_nominatim(addr)
            if coords:
                cache[rid] = coords
                nuevos += 1
                print(f"  [{rid:>3}] ✓ {r['nombre'][:40]:<40} → {coords[0]}, {coords[1]}")
            else:
                cache[rid] = None
                print(f"  [{rid:>3}] ✗ {r['nombre'][:40]:<40} — sin resultado")
            time.sleep(1.1)  # Nominatim rate limit: 1 req/s
        _guardar_cache_coords(cache)
        print(f"  {nuevos}/{len(pendientes)} geocodificados. Guardados en .cache_coords.json\n")
    else:
        print(f"Coordenadas: {len(cache)} en caché, sin pendientes.\n")

    # Asignar coords a cada record desde la caché
    for r in records:
        rid = str(r['id_restaurante'])
        r['coords'] = cache.get(rid)  # None si no se pudo geocodificar

    return records

print(f"Leyendo {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

NUEVAS_DIMS = ['ninos','mascotas','terraza','vistas','musica_directo','romantico']
DIMS_BASE   = ['ambiente','ruido','precio','servicio','velocidad','limpieza']
TODAS_DIMS  = DIMS_BASE + NUEVAS_DIMS

cols_siempre = [
    'id_restaurante','nombre','direccion','valoracion_google','pct_positivo',
    'top5_platos','terminos_tfidf','avg_estrellas_modelo','n_resenas',
]
# Columnas opcionales — solo si existen en el CSV
_CRITERIOS_COLS = ['ninos','mascotas','terraza','vistas','musica_directo','romantico']
cols_opcionales = (
    ['todos_platos']
    + [f'criterio_{c}' for c in _CRITERIOS_COLS]
    + [f'criterio_{c}_frases' for c in _CRITERIOS_COLS]
    + ['servicio_frases']
)
cols_base = cols_siempre + [c for c in cols_opcionales if c in df.columns]
cols_dims = []
for d in TODAS_DIMS:
    for suf in ['_menciones','_pos','_neg','_avg_stars']:
        cols_dims.append(d + suf)

# Solo incluir columnas que existan en el CSV (compatibilidad hacia atrás)
cols_disponibles = set(df.columns)
cols = cols_base + [c for c in cols_dims if c in cols_disponibles]

df2 = df[cols].copy()
for c in ['top5_platos','terminos_tfidf','direccion']:
    df2[c] = df2[c].fillna('')
# Columnas opcionales — rellenar solo si existen
for c in cols_opcionales:
    if c in df2.columns:
        if 'criterio_' in c and '_frases' not in c:
            df2[c] = df2[c].fillna(False)
        else:
            df2[c] = df2[c].fillna('')
for c in [c for c in cols_dims if c in cols_disponibles]:
    df2[c] = pd.to_numeric(df2[c], errors='coerce').fillna(0)

records = df2.to_dict('records')

# Geocodificar direcciones con Nominatim (usa caché .cache_coords.json)
records = geocodificar_restaurantes(records)

# Calcular dist_sol para cada restaurante
for r in records:
    if r['coords']:
        r['dist_sol'] = round(haversine_py(SOL_COORDS, r['coords']), 2)
    else:
        r['dist_sol'] = None

    def _score(dim, min_m=2, default=5.0):
        """Score 0-10 para una dimensión basado en avg_stars, menciones y negativas."""
        m   = r.get(f'{dim}_menciones', 0)
        avg = r.get(f'{dim}_avg_stars', 0)
        neg = r.get(f'{dim}_neg', 0)
        if m < min_m or avg == 0:
            return default
        return round(min(10, (avg / 5) * 10 * (1 - neg / (m + 1))), 1)

    def _bool_dim(dim, umbral_menciones=2):
        """True si la dimensión tiene menciones suficientes (señal en reseñas)."""
        return r.get(f'{dim}_menciones', 0) >= umbral_menciones

    r['score_ambiente']  = _score('ambiente')
    r['score_tranquilo'] = _score('ruido', default=6.0)
    r['score_precio']    = _score('precio')

    # Criterios cualitativos — leer columnas criterio_* generadas por Gemini.
    # Si el CSV no las tiene aún (ejecución anterior), caer a heurística de menciones.
    for c in ['ninos','mascotas','terraza','vistas','musica_directo','romantico']:
        col = f'criterio_{c}'
        if col in r:
            val = r[col]
            if isinstance(val, str):
                val = val.strip().lower() == 'true'
            r[f'flag_{c}'] = bool(val)
        else:
            r[f'flag_{c}'] = _bool_dim(c, umbral_menciones=2)

con_coords = sum(1 for r in records if r['coords'])
print(f"  {len(records)} restaurantes cargados ({con_coords} con coordenadas).")

DATA_JS = f"const RESTAURANTES = {json.dumps(records, ensure_ascii=False, default=str)};"

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0f0f0f;
  --surface:   #161616;
  --surface2:  #1e1e1e;
  --border:    #2a2a2a;
  --border2:   #222;
  --text:      #f0ece4;
  --text2:     #888;
  --text3:     #444;
  --accent:    #c8a96e;
  --accent-bg: #1a1505;
  --accent-dim:#c8a96e22;
  --green:     #4caf82;
  --green-bg:  #0a1f14;
  --amber:     #c8963e;
  --amber-bg:  #1e1608;
  --blue:      #5a8aaa;
  --blue-bg:   #08151e;
  --tag-bg:    #1a1a1a;
  --shadow:    0 1px 8px rgba(0,0,0,.4), 0 4px 20px rgba(0,0,0,.3);
  --radius:    12px;
}

body {
  font-family: 'DM Sans', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 0;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }

/* ── Layout ── */
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 700px;
  margin: 0 auto;
}

/* ── Header ── */
.header {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.header-inner { display: flex; align-items: center; gap: 12px; }
.logo-wrap {
  width: 42px; height: 42px; border-radius: 10px;
  background: var(--accent-bg);
  border: 1px solid var(--accent-dim);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
}
.header h1 {
  font-family: 'DM Serif Display', serif;
  font-size: 18px; color: var(--text);
  letter-spacing: -0.3px; font-weight: 400;
}
.header .sub {
  font-size: 10px; color: var(--text3);
  letter-spacing: 0.8px; text-transform: uppercase; margin-top: 2px;
}
.btn-nuevo {
  background: transparent; border: 1px solid var(--border);
  color: var(--text2); border-radius: 20px; padding: 5px 14px;
  font-size: 12px; cursor: pointer; font-family: inherit;
  transition: border-color .15s, color .15s;
}
.btn-nuevo:hover { border-color: var(--accent); color: var(--accent); }

/* ── Scrollable results area ── */
.results-area {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* ── Search footer ── */
.footer {
  padding: 10px 16px 16px;
  border-top: 1px solid var(--border);
  background: var(--bg);
}
.input-wrap {
  display: flex; gap: 8px; align-items: center;
  background: var(--surface);
  border-radius: 28px;
  padding: 4px 4px 4px 16px;
  border: 1px solid var(--border);
  transition: border-color .15s;
}
.input-wrap:focus-within { border-color: var(--accent); }
#query {
  flex: 1; background: transparent; border: none;
  color: var(--text); font-size: 14px; outline: none;
  padding: 8px 0; font-family: inherit;
}
#query::placeholder { color: var(--text3); }
.btn-search {
  background: var(--accent); border: none; border-radius: 50%;
  width: 38px; height: 38px; font-size: 16px; cursor: pointer;
  color: #111; font-weight: bold; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  transition: transform .1s, opacity .15s;
}
.btn-search:hover { opacity: .85; }
.btn-search:active { transform: scale(.93); }
.hint { font-size: 11px; color: var(--text3); text-align: center; margin-top: 8px; }

/* ── Filters ── */
.filters-section {
  background: var(--bg);
  padding: 0 0 8px;
}
.filters-title {
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .1em;
  color: var(--text3); margin-bottom: 8px;
}
.filters-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.filter-chip {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 12px; font-weight: 500; padding: 6px 13px;
  border: 1px solid var(--border); border-radius: 100px;
  cursor: pointer; background: transparent; color: var(--text2);
  transition: all .15s; user-select: none; white-space: nowrap;
  font-family: inherit;
}
.filter-chip:hover { border-color: var(--accent); color: var(--accent); }
.filter-chip.active { border-color: var(--accent); background: var(--accent); color: #111; font-weight: 600; }
.filter-chip .icon { font-size: 13px; line-height: 1; }

/* ── Status ── */
.status-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 0 8px; flex-wrap: wrap; gap: 6px;
}
.status { font-size: 12px; color: var(--text3); }
.active-filters { display: flex; flex-wrap: wrap; gap: 5px; }
.af-badge {
  font-size: 11px; padding: 3px 10px;
  background: var(--accent-bg); color: var(--accent);
  border-radius: 100px; font-weight: 500;
  border: 1px solid var(--accent-dim);
}

/* ── Welcome / empty state ── */
.welcome {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 2rem 1rem; gap: 10px; text-align: center;
}
.welcome-icon { font-size: 36px; margin-bottom: 8px; }
.welcome h2 {
  font-family: 'DM Serif Display', serif;
  font-size: 22px; font-weight: 400; color: var(--text);
  letter-spacing: -0.3px;
}
.welcome p { font-size: 13px; color: var(--text2); max-width: 300px; line-height: 1.6; }
.sugs { display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; margin-top: 10px; }
.sug {
  background: transparent; border: 1px solid var(--border);
  color: var(--text2); border-radius: 20px; padding: 6px 14px;
  font-size: 12px; cursor: pointer; font-family: inherit;
  transition: border-color .15s, color .15s;
}
.sug:hover { border-color: var(--accent); color: var(--accent); }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
  cursor: pointer;
  transition: border-color .15s, background .15s;
  position: relative;
  overflow: hidden;
}
.card:hover { border-color: var(--accent-dim); background: #1a1a1a; }
.card::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px; background: transparent;
  transition: background .15s;
}
.card:hover::before { background: var(--accent); }

.card-top {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 12px; margin-bottom: 10px;
}
.card-index {
  font-family: 'DM Serif Display', Georgia, serif;
  font-size: 20px; color: var(--text3);
  line-height: 1; flex-shrink: 0; margin-top: 2px;
}
.card-info { flex: 1; min-width: 0; }
.card-name {
  font-size: 15px; font-weight: 600; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.card-addr {
  font-size: 11px; color: var(--text3); margin-top: 3px;
  display: flex; align-items: center; gap: 4px;
}
.card-right {
  display: flex; flex-direction: column;
  align-items: flex-end; gap: 4px; flex-shrink: 0;
}
.badge {
  font-size: 11px; padding: 3px 9px;
  border-radius: 100px; font-weight: 600;
}
.badge-green { background: var(--green-bg); color: var(--green); border: 1px solid #0a2a1a; }
.badge-amber { background: var(--amber-bg); color: var(--amber); border: 1px solid #2a1e08; }
.badge-red   { background: #1e0a0a; color: #c04040; border: 1px solid #2a0e0e; }
.stars { font-size: 12px; color: var(--accent); font-weight: 500; }
.maps-link {
  font-size: 11px; color: var(--blue); text-decoration: none;
  display: flex; align-items: center; gap: 3px;
}
.maps-link:hover { color: var(--accent); }

/* ── Criteria badges ── */
.criteria-badges { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }
.crit-badge {
  font-size: 11px; padding: 3px 9px; border-radius: 100px;
  font-weight: 500; display: inline-flex; align-items: center; gap: 3px;
  border: 1px solid transparent;
}
.crit-badge.good   { background: var(--green-bg); color: var(--green); border-color: #0a2a1a; }
.crit-badge.warn   { background: var(--amber-bg); color: var(--amber); border-color: #2a1e08; }
.crit-badge.info   { background: var(--blue-bg);  color: var(--blue);  border-color: #0a1e2a; }
.crit-badge.neutral{ background: #1a1a1a; color: var(--text2); border-color: var(--border); }

/* ── Reason box ── */
.razon {
  font-size: 12px; color: var(--text2); margin-top: 6px;
  padding: 6px 10px; background: #0f0f0f;
  border-radius: 8px; border-left: 2px solid var(--accent);
  line-height: 1.4;
}
.razon strong { color: var(--accent); font-weight: 600; }

/* ── Platos ── */
.platos { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
.plato {
  font-size: 11px; padding: 3px 9px;
  background: var(--tag-bg); border-radius: 100px; color: var(--text2);
  border: 1px solid var(--border2);
}
.plato.match {
  background: #0e1a2a; color: #7ab3d4;
  border: 1px solid #1a3a5a; font-weight: 600;
}
.plato-menciones {
  font-size: 10px; font-weight: 700;
  background: var(--accent); color: #111;
  padding: 1px 5px; border-radius: 8px; margin-left: 3px;
  vertical-align: middle;
}

.dist { font-size: 11px; color: var(--text3); margin-top: 6px; display: flex; align-items: center; gap: 4px; }

/* ── Modal ── */
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,.85);
  z-index: 100; display: flex; align-items: flex-end;
  justify-content: center; padding: 0;
  animation: fadeIn .2s ease;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.modal {
  background: #111; border: 1px solid var(--border);
  border-radius: 20px 20px 0 0; width: 100%; max-width: 700px;
  max-height: 88vh; overflow-y: auto;
  padding: 20px 20px 32px;
  animation: slideUp .25s ease;
}
@keyframes slideUp { from { transform: translateY(40px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
.modal-handle {
  width: 40px; height: 4px; background: var(--border);
  border-radius: 2px; margin: 0 auto 16px;
}
.modal-header {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 12px; margin-bottom: 14px;
}
.modal-title {
  font-family: 'DM Serif Display', serif;
  font-size: 20px; font-weight: 400; color: var(--text);
}
.modal-addr { font-size: 12px; color: var(--text3); margin-top: 4px; }
.modal-close {
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text2); border-radius: 50%; width: 32px; height: 32px;
  cursor: pointer; font-size: 16px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
}
.modal-section {
  background: var(--surface); border-radius: 10px;
  padding: 12px 14px; margin-bottom: 10px;
  border: 1px solid var(--border);
}
.modal-section-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text3); margin-bottom: 8px;
}
.modal-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.modal-badge {
  font-size: 12px; padding: 4px 12px;
  background: var(--accent-bg); color: var(--accent);
  border: 1px solid var(--accent-dim); border-radius: 100px;
}

.empty {
  text-align: center; padding: 3rem 2rem; color: var(--text3);
  font-size: 13px;
}
.empty .e-icon { font-size: 28px; margin-bottom: .5rem; opacity: .4; }

@media(max-width:480px){
  .modal { border-radius: 16px 16px 0 0; }
}
"""

JS = r"""
// ─── Data & state ────────────────────────────────────────────────
const _geoCache = {};
const SOL = [40.4168, -3.7038]; // Puerta del Sol

const CRITERIOS = {
  buen_ambiente:  { label: 'Buen ambiente',      icon: '✨' },
  tranquilo:      { label: 'Tranquilo',           icon: '🤫' },
  ruidoso:        { label: 'Con marcha',          icon: '🎉' },
  precio_ok:      { label: 'Buen precio',         icon: '💶' },
  centrico:       { label: 'Céntrico',             icon: '📍' },
  muy_valorado:   { label: 'Muy valorado',        icon: '⭐' },
  ninos:          { label: 'Apto niños',          icon: '👶' },
  mascotas:       { label: 'Admite mascotas',     icon: '🐾' },
  terraza:        { label: 'Terraza',             icon: '☀️' },
  vistas:         { label: 'Con vistas',          icon: '🏙️' },
  musica_directo: { label: 'Música en directo',   icon: '🎵' },
  romantico:      { label: 'Romántico',           icon: '🕯️' },
};

let activeFilters = new Set();

// ─── Geo utils ───────────────────────────────────────────────────
// Zonas y landmarks de Madrid — funciona sin red (file://)
const ZONAS_MADRID = {
  "santiago bernabeu":[40.4531,-3.6884],"bernabeu":[40.4531,-3.6884],
  "estadio bernabeu":[40.4531,-3.6884],"estadio santiago bernabeu":[40.4531,-3.6884],
  "wanda metropolitano":[40.4362,-3.5997],"metropolitano":[40.4362,-3.5997],
  "sol":[40.4168,-3.7038],"puerta del sol":[40.4168,-3.7038],
  "gran via":[40.4200,-3.7040],"gran vía":[40.4200,-3.7040],
  "chueca":[40.4237,-3.6979],"malasana":[40.4260,-3.7060],"malasaña":[40.4260,-3.7060],
  "lavapies":[40.4078,-3.7018],"lavapiés":[40.4078,-3.7018],
  "la latina":[40.4110,-3.7092],"latina":[40.4110,-3.7092],
  "huertas":[40.4131,-3.6978],"letras":[40.4131,-3.6978],
  "embajadores":[40.4060,-3.7050],"tribunal":[40.4265,-3.6998],
  "alonso martinez":[40.4265,-3.6935],"alonso martínez":[40.4265,-3.6935],
  "colon":[40.4238,-3.6888],"colón":[40.4238,-3.6888],
  "recoletos":[40.4238,-3.6898],"retiro":[40.4153,-3.6844],
  "parque del retiro":[40.4153,-3.6844],"parque retiro":[40.4153,-3.6844],
  "salamanca":[40.4298,-3.6831],"serrano":[40.4298,-3.6831],
  "goya":[40.4248,-3.6788],"jorge juan":[40.4248,-3.6748],
  "velazquez":[40.4298,-3.6831],"velázquez":[40.4298,-3.6831],
  "chamberi":[40.4350,-3.7000],"chamberí":[40.4350,-3.7000],
  "almagro":[40.4330,-3.6930],"arguelles":[40.4268,-3.7168],"argüelles":[40.4268,-3.7168],
  "moncloa":[40.4349,-3.7189],"principe pio":[40.4175,-3.7200],"príncipe pío":[40.4175,-3.7200],
  "opera":[40.4185,-3.7118],"ópera":[40.4185,-3.7118],
  "palacio real":[40.4179,-3.7143],"palacio":[40.4179,-3.7143],
  "tetuan":[40.4620,-3.6980],"tetuán":[40.4620,-3.6980],
  "cuatro caminos":[40.4456,-3.7010],"nuevos ministerios":[40.4488,-3.6922],
  "castellana":[40.4390,-3.6880],"paseo castellana":[40.4390,-3.6880],
  "azca":[40.4530,-3.6940],"hortaleza":[40.4780,-3.6540],
  "arturo soria":[40.4510,-3.6470],"ciudad lineal":[40.4510,-3.6470],
  "vallecas":[40.3878,-3.6580],"usera":[40.3878,-3.7118],
  "carabanchel":[40.3878,-3.7388],"mostoles":[40.3220,-3.8640],"mostóles":[40.3220,-3.8640],
  "leganes":[40.3280,-3.7640],"leganés":[40.3280,-3.7640],
  "plaza mayor":[40.4154,-3.7074],"mercado san miguel":[40.4154,-3.7074],
  "san miguel":[40.4154,-3.7074],"prado":[40.4138,-3.6921],
  "museo del prado":[40.4138,-3.6921],"thyssen":[40.4158,-3.6948],
  "reina sofia":[40.4082,-3.6944],"reina sofía":[40.4082,-3.6944],
  "atocha":[40.4072,-3.6898],"chamartin":[40.4723,-3.6847],"chamartín":[40.4723,-3.6847],
  "ifema":[40.4700,-3.6070],"aeropuerto":[40.4983,-3.5676],"barajas":[40.4983,-3.5676],
  "fuencarral":[40.4265,-3.7018],"plaza espana":[40.4238,-3.7148],
  "plaza españa":[40.4238,-3.7148],"callao":[40.4208,-3.7058],
  "centro":[40.4168,-3.7038],"centro madrid":[40.4168,-3.7038]
};

async function geocodificarUsuario(lugar) {
  const key = norm(lugar);
  if (_geoCache[key]) return _geoCache[key];

  // 1. Diccionario local — instantáneo, funciona en file://
  if (ZONAS_MADRID[key]) { _geoCache[key] = ZONAS_MADRID[key]; return ZONAS_MADRID[key]; }
  for (const [zona, coords] of Object.entries(ZONAS_MADRID)) {
    if (key.includes(zona) || zona.includes(key)) {
      _geoCache[key] = coords; return coords;
    }
  }

  // 2. Nominatim (solo funciona desde http://, no desde file://)
  try {
    const q = encodeURIComponent(lugar + ', Madrid, España');
    const url = 'https://nominatim.openstreetmap.org/search?q=' + q + '&format=json&limit=1&countrycodes=es';
    const res = await fetch(url, {headers:{'Accept-Language':'es','User-Agent':'agente-restaurantes-madrid'}});
    const data = await res.json();
    if (data.length > 0) {
      const coords = [parseFloat(data[0].lat), parseFloat(data[0].lon)];
      _geoCache[key] = coords;
      return coords;
    }
  } catch(e) { /* silencioso en file:// */ }
  return null;
}

function generarRazon(r, s, locText, activeFiltersArr) {
  // Construye una frase explicando por qué este restaurante sale primero
  const partes = [];

  // 1. Distancia si hay zona buscada
  if (s.dk != null) {
    partes.push(`a <strong>${s.dk.toFixed(2)} km</strong> de ${locText}`);
  }

  // 2. Por qué cumple los criterios activos
  const CRITERIO_RAZONES = {
    romantico:      () => r.criterio_romantico_frases
                          ? `ambiente romántico — "${r.criterio_romantico_frases.split('|')[0].trim().substring(0,70)}"`
                          : 'mencionado como romántico en reseñas',
    ninos:          () => r.criterio_ninos_frases
                          ? `apto para niños — "${r.criterio_ninos_frases.split('|')[0].trim().substring(0,70)}"`
                          : 'confirmado apto para niños',
    mascotas:       () => r.criterio_mascotas_frases
                          ? `admite mascotas — "${r.criterio_mascotas_frases.split('|')[0].trim().substring(0,70)}"`
                          : 'admite mascotas según reseñas',
    terraza:        () => r.criterio_terraza_frases
                          ? `tiene terraza — "${r.criterio_terraza_frases.split('|')[0].trim().substring(0,70)}"`
                          : 'con terraza confirmada',
    vistas:         () => r.criterio_vistas_frases
                          ? `con vistas — "${r.criterio_vistas_frases.split('|')[0].trim().substring(0,70)}"`
                          : 'con vistas según reseñas',
    musica_directo: () => r.criterio_musica_directo_frases
                          ? `música en directo — "${r.criterio_musica_directo_frases.split('|')[0].trim().substring(0,70)}"`
                          : 'con música en directo',
    tranquilo:      () => `ambiente tranquilo (${r.ruido_neg === 0 ? 'sin quejas de ruido' : 'pocas quejas'})`,
    precio_ok:      () => `buen precio (★${r.precio_avg_stars || ''} en reseñas de precio)`,
    buen_ambiente:  () => `gran ambiente (★${r.ambiente_avg_stars || ''} en reseñas de ambiente)`,
    ruidoso:        () => `ambiente animado (${r.ambiente_pos} menciones positivas)`,
    centrico:       () => `muy céntrico (${r.dist_sol} km del centro)`,
    muy_valorado:   () => `${r.pct_positivo}% reseñas positivas · ★${r.valoracion_google}`,
  };

  for (const f of activeFiltersArr) {
    if (CRITERIO_RAZONES[f]) {
      partes.push(CRITERIO_RAZONES[f]());
      break; // una razón de criterio es suficiente
    }
  }

  // 3. Si hay match de plato o cocina, mencionarlo
  if (s.cocina) {
    const platosMatch = s.mp.slice(0, 2).map(p => p.nombre).join(', ');
    if (platosMatch) partes.push(`${COCINAS[s.cocina].label} — ${platosMatch}`);
    else partes.push(COCINAS[s.cocina].label);
  } else if (s.mp && s.mp.length > 0) {
    const p = s.mp[0];
    partes.push(`${p.menciones} reseñas destacan <strong>${p.nombre}</strong>`);
  }

  // 4. Puntuación general como cierre si no hay otras razones
  if (partes.length === 0) {
    partes.push(`${r.pct_positivo}% reseñas positivas · ★${r.valoracion_google}`);
  }

  return partes.join(' · ');
}

function haversine(a, b) {
  const R = 6371;
  const dLat = (b[0]-a[0]) * Math.PI/180;
  const dLon = (b[1]-a[1]) * Math.PI/180;
  const f = Math.sin(dLat/2)**2 + Math.cos(a[0]*Math.PI/180)*Math.cos(b[0]*Math.PI/180)*Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(f), Math.sqrt(1-f));
}

// ─── Text utils ──────────────────────────────────────────────────
function norm(s) {
  return s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'');
}
function tokenize(s) {
  return norm(s).split(/[\s,()\/\-]+/).filter(t => t.length > 1);
}
function extractLocationText(q) {
  const n = norm(q);
  // Patrones con artículo: "cerca del bernabeu", "cerca de la latina", "en el retiro"
  const patterns = [
    /cerca del? (.+)$/i,
    /cerca de la (.+)$/i,
    /cerca de los (.+)$/i,
    /cerca de las (.+)$/i,
    /cerca de (.+)$/i,
    /en el (.+)$/i,
    /en la (.+)$/i,
    /en los (.+)$/i,
    /en las (.+)$/i,
    /en (.+)$/i,
    /desde (.+)$/i,
    /por (.+)$/i,
    /junto al? (.+)$/i,
    /junto a (.+)$/i,
    /al lado del? (.+)$/i,
    /al lado de (.+)$/i,
    /pr[oó]ximo al? (.+)$/i,
    /pr[oó]ximo a (.+)$/i,
  ];
  for (const p of patterns) {
    const m = n.match(p);
    if (m) {
      // Limpiar palabras sueltas que no son zona ("un", "restaurante", etc.)
      const loc = m[1].trim()
        .replace(/^(un |una |el |la |los |las |este |esta )/i, '')
        .trim();
      if (loc.length >= 3) return loc;
    }
  }
  return null;
}
function parseListaPlatos(str) {
  // Parsea "plato1(n), plato2(m)" → [{nombre, menciones}]
  if (!str) return [];
  return str.split(',').map(p => {
    const m = p.trim().match(/^(.+?)\((\d+)\)$/);
    if (m) return { nombre: m[1].trim(), menciones: parseInt(m[2]) };
    const nm = p.trim();
    return nm ? { nombre: nm, menciones: 1 } : null;
  }).filter(Boolean);
}

// ─── Categorías de cocina ────────────────────────────────────────
// Mapea nombre de cocina → platos representativos (en norm sin tildes)
const COCINAS = {
  mexicana:   { label: 'Cocina mexicana',  platos: ['tacos','burrito','quesadilla','fajitas','guacamole','nachos','enchilada','pozole','carnitas','mole','tostada','chile'] },
  italiana:   { label: 'Cocina italiana',  platos: ['pasta','pizza','risotto','lasana','lasaña','carbonara','tiramisu','tiramisú','penne','ravioli','tagliatelle','gnocchi','focaccia','bruschetta'] },
  japonesa:   { label: 'Cocina japonesa',  platos: ['sushi','sashimi','ramen','gyozas','edamame','tempura','miso','udon','yakitori','takoyaki','onigiri','mochi','katsu'] },
  india:      { label: 'Cocina india',     platos: ['tikka','masala','biryani','naan','samosa','curry','korma','dal','tandoori','chapati','lassi','pakora','chana'] },
  peruana:    { label: 'Cocina peruana',   platos: ['ceviche','tiradito','lomo saltado','causa','anticuchos','leche de tigre','aji de gallina','chicharron','arroz con leche','picarones'] },
  española:   { label: 'Cocina española',  platos: ['paella','tortilla','croquetas','gazpacho','salmorejo','jamon','jamón','cocido','fabada','pulpo','patatas bravas','pisto','rabo','chuleton','chuletón'] },
  asturiana:  { label: 'Cocina asturiana', platos: ['cachopo','fabada','cachopo','sidra','morcilla','oricios','pote','pixin'] },
  gallega:    { label: 'Cocina gallega',   platos: ['pulpo','empanada','caldo gallego','lacón','percebes','mejillones','navajas','zamburiñas','berberechos'] },
  vasca:      { label: 'Cocina vasca',     platos: ['pintxos','pintxo','gilda','bacalao','txangurro','marmitako','kokotxas','pil pil','txakoli'] },
  francesa:   { label: 'Cocina francesa',  platos: ['foie','confit','ratatouille','crepe','croissant','soufflé','boeuf','tarte','coq au vin','bouillabaisse'] },
  griega:     { label: 'Cocina griega',    platos: ['hummus','gyros','souvlaki','falafel','tzatziki','moussaka','spanakopita','baklava','dolmades'] },
  arabe:      { label: 'Cocina árabe',     platos: ['hummus','falafel','kebab','shawarma','tabule','cuscus','baklava','pita','labneh','fattoush'] },
  venezolana: { label: 'Cocina venezolana',platos: ['arepa','pabellon','pabellón','cachapa','tequeño','tequeno','hallaca','pernil','mandoca','caraotas'] },
  colombiana: { label: 'Cocina colombiana',platos: ['bandeja paisa','ajiaco','empanada','arepa','sancocho','arroz con pollo','buñuelo','changua'] },
  china:      { label: 'Cocina china',     platos: ['dim sum','wonton','pato pekines','chow mein','arroz frito','spring roll','mapo tofu','baozi','dumplings'] },
  tailandesa: { label: 'Cocina tailandesa',platos: ['pad thai','curry verde','curry rojo','tom yum','som tam','massaman','satay','mango sticky rice'] },
  americana:  { label: 'Cocina americana', platos: ['hamburguesa','hamburger','burger','costillas','costillar','pulled pork','mac cheese','cheesecake','brownie','bbq','barbacoa'] },
  mediterranea:{ label: 'Cocina mediterránea', platos: ['hummus','falafel','pita','baba ganoush','tabulé','couscous','dolmades','labneh','shakshuka'] },
};

// Sinónimos de cocina que el usuario puede escribir
const SINONIMOS_COCINA = {
  'mexicano': 'mexicana', 'mexico': 'mexicana', 'méjico': 'mexicana', 'mejico': 'mexicana',
  'italiano': 'italiana', 'italia': 'italiana',
  'japones': 'japonesa', 'japonés': 'japonesa', 'japon': 'japonesa', 'japón': 'japonesa', 'japones': 'japonesa',
  'indio': 'india', 'hindu': 'india', 'hindú': 'india', 'india': 'india',
  'peruano': 'peruana', 'peru': 'peruana', 'perú': 'peruana',
  'español': 'española', 'espanola': 'española', 'espanol': 'española',
  'asturiano': 'asturiana', 'asturias': 'asturiana',
  'gallego': 'gallega', 'galicia': 'gallega',
  'vasco': 'vasca', 'pais vasco': 'vasca', 'euskadi': 'vasca', 'basque': 'vasca',
  'frances': 'francesa', 'francés': 'francesa', 'francia': 'francesa',
  'griego': 'griega', 'grecia': 'griega',
  'arabe': 'arabe', 'árabe': 'arabe', 'libanes': 'arabe', 'libanés': 'arabe', 'libano': 'arabe',
  'venezolano': 'venezolana', 'venezuela': 'venezolana',
  'colombiano': 'colombiana', 'colombia': 'colombiana',
  'chino': 'china', 'china': 'china',
  'tailandes': 'tailandesa', 'tailandés': 'tailandesa', 'tailandia': 'tailandesa', 'thai': 'tailandesa',
  'americano': 'americana', 'americano': 'americana', 'usa': 'americana',
  'mediterraneo': 'mediterranea', 'mediterráneo': 'mediterranea',
};

function detectarCocina(query) {
  const qn = norm(query);
  for (const [sinonimo, cocina] of Object.entries(SINONIMOS_COCINA)) {
    // Palabra completa — evitar substring matches
    const padded = ' ' + qn + ' ';
    if (padded.includes(' ' + sinonimo + ' ') || qn === sinonimo || qn.startsWith(sinonimo + ' ') || qn.endsWith(' ' + sinonimo)) {
      return cocina;
    }
  }
  return null;
}

function scorePorCocina(r, cocina) {
  // Cuenta cuántos platos de la cocina tiene el restaurante en su lista
  const platos = COCINAS[cocina].platos;
  const fuente = norm((r.todos_platos || r.top5_platos || '') + ' ' + r.terminos_tfidf);
  const platosList = parseListaPlatos(r.todos_platos || r.top5_platos || '');

  let ms = 0, mp = [];
  for (const plato of platos) {
    if (fuente.includes(plato)) {
      ms += 2;
      const po = platosList.find(p => norm(p.nombre).includes(plato) || plato.includes(norm(p.nombre)));
      if (po && !mp.find(x => x.nombre === po.nombre)) {
        mp.push(po);
        if (po.menciones > 1) ms += Math.log2(po.menciones);
      }
    }
  }
  return ms > 0 ? {r, ms, mp, cocina} : null;
}

function scoreRestaurante(r, qTokens) {
  // Buscar en top5_platos + todos_platos + terminos_tfidf + nombre
  const fuentePlatos = r.todos_platos || r.top5_platos || '';
  const txt = norm(fuentePlatos + ' ' + r.terminos_tfidf + ' ' + r.nombre);
  const pts = tokenize(txt);
  let ms = 0, mp = [];
  for (const qt of qTokens) {
    if (qt.length < 3) continue;
    for (const pt of pts) {
      if (pt.includes(qt) || qt.includes(pt)) {
        ms += pt === qt ? 2 : 1;
        // Buscar en todos_platos primero para obtener menciones reales
        const fuente = parseListaPlatos(fuentePlatos);
        const po = fuente.find(p => norm(p.nombre).includes(qt) || qt.includes(norm(p.nombre)));
        if (po && !mp.find(x => x.nombre === po.nombre)) mp.push(po);
        // Sumar menciones del plato al score — más menciones = más relevante
        if (po && po.menciones > 1) ms += Math.log2(po.menciones);
        break;
      }
    }
  }
  // ms final: base (tokens) + log de menciones del plato buscado
  return ms > 0 ? {r, ms, mp} : null;
}

// ─── Criteria scoring ────────────────────────────────────────────
function pasaCriterio(r, filtro) {
  switch(filtro) {
    case 'buen_ambiente':
      return r.score_ambiente >= 7.5;
    case 'tranquilo':
      return r.score_tranquilo >= 7.0 && r.ruido_neg === 0;
    case 'ruidoso':
      return r.ambiente_pos >= 20 || (r.ruido_pos > r.ruido_neg && r.ruido_menciones >= 3);
    case 'precio_ok':
      return r.score_precio >= 7.0;
    case 'centrico':
      return r.dist_sol !== null && r.dist_sol <= 1.5;
    case 'muy_valorado':
      return r.pct_positivo >= 90 && r.valoracion_google >= 4.7;
    // ── Nuevos criterios — basados en detección directa en reseñas ──
    case 'ninos':
      return r.flag_ninos;
    case 'mascotas':
      return r.flag_mascotas;
    case 'terraza':
      return r.flag_terraza;
    case 'vistas':
      return r.flag_vistas;
    case 'musica_directo':
      return r.flag_musica_directo;
    case 'romantico':
      return r.flag_romantico;
    default: return true;
  }
}

function getCriteriaBadgesHtml(r) {
  const badges = [];
  if (r.score_ambiente >= 8) badges.push({cls:'good', icon:'✨', text:'Gran ambiente'});
  else if (r.score_ambiente >= 7) badges.push({cls:'good', icon:'✨', text:'Buen ambiente'});

  if (r.ruido_neg === 0 && r.score_tranquilo >= 7.5) badges.push({cls:'info', icon:'🤫', text:'Tranquilo'});
  else if (r.ruido_neg > 0) badges.push({cls:'warn', icon:'🎉', text:'Animado'});

  if (r.score_precio >= 8) badges.push({cls:'good', icon:'💶', text:'Muy buen precio'});
  else if (r.score_precio >= 7) badges.push({cls:'info', icon:'💶', text:'Precio razonable'});

  if (r.dist_sol !== null && r.dist_sol <= 0.8) badges.push({cls:'info', icon:'📍', text:'Muy céntrico'});
  else if (r.dist_sol !== null && r.dist_sol <= 1.5) badges.push({cls:'neutral', icon:'📍', text:'Céntrico'});

  if (r.pct_positivo >= 95) badges.push({cls:'good', icon:'⭐', text:'Top reseñas'});

  if (r.flag_terraza) badges.push({cls:'info', icon:'☀️',  text:'Terraza',    frase: r.criterio_terraza_frases||''});
  if (r.flag_vistas)  badges.push({cls:'info', icon:'🏙️', text:'Con vistas',  frase: r.criterio_vistas_frases||''});
  if (r.flag_ninos)    badges.push({cls:'neutral', icon:'👶', text:'Apto niños',    frase: r.criterio_ninos_frases||''});
  if (r.flag_mascotas) badges.push({cls:'neutral', icon:'🐾', text:'Mascotas',      frase: r.criterio_mascotas_frases||''});
  if (r.flag_musica_directo)  badges.push({cls:'neutral', icon:'🎵', text:'Música en directo', frase: r.criterio_musica_directo_frases||''});
  if (r.flag_romantico)       badges.push({cls:'neutral', icon:'🕯️', text:'Romántico',         frase: r.criterio_romantico_frases||''});

  return badges.slice(0, 5).map(b => {
    const tooltip = b.frase ? ` title="${b.frase.replace(/"/g,"'").substring(0,120)}"` : '';
    return `<span class="crit-badge ${b.cls}"${tooltip}>${b.icon} ${b.text}</span>`;
  }).join('');
}

// ─── Filter chips ─────────────────────────────────────────────────
function toggleFilter(key) {
  // ruidoso y tranquilo se excluyen mutuamente
  if (key === 'ruidoso' && activeFilters.has('tranquilo')) activeFilters.delete('tranquilo');
  if (key === 'tranquilo' && activeFilters.has('ruidoso')) activeFilters.delete('ruidoso');

  if (activeFilters.has(key)) {
    activeFilters.delete(key);
  } else {
    activeFilters.add(key);
  }
  renderFilterChips();
  buscar();
}

function renderFilterChips() {
  Object.keys(CRITERIOS).forEach(key => {
    const el = document.getElementById('fc-' + key);
    if (el) el.classList.toggle('active', activeFilters.has(key));
  });
}


// ─── Main search ─────────────────────────────────────────────────
async function buscar() {
  const query   = document.getElementById('query').value.trim();
  const status  = document.getElementById('status');
  const results = document.getElementById('results');
  const afRow   = document.getElementById('active-filters');
  results.innerHTML = '';
  results.style.display = 'none';

  // Show active filter badges
  afRow.innerHTML = [...activeFilters].map(k =>
    `<span class="af-badge">${CRITERIOS[k].icon} ${CRITERIOS[k].label}</span>`
  ).join('');

  const hasQuery   = query.length > 0;
  const hasFilters = activeFilters.size > 0;

  if (!hasQuery && !hasFilters) {
    status.textContent = 'Escribe un plato, zona o selecciona criterios';
    return;
  }

  // ── Detección de intención — mapear frases a criterios automáticamente ─────
  // Si la query no tiene plato ni zona reconocibles, buscar intención
  const INTENCIONES = [
    { palabras: [
        'pareja','romantico','romantica','romántico','romántica',
        'intimo','íntimo','intimidad','cena romantica','cena romántica',
        'aniversario','fecha','primera cita','san valentin','san valentín',
        'pedida','compromiso','declarar','declaracion','declaración',
        'sorprender','sorpresa','ocasion especial','ocasión especial',
        'noche especial','momento especial','algo especial','muy especial',
        'cumpleanos','cumpleaños','celebracion','celebración','celebrar',
        'ocasion importante','ocasión importante','algo diferente',
        'sitio especial','lugar especial','restaurante especial',
        'cena especial','cena intima','cena íntima',
        'para dos','solo nosotros','velada','velada especial',
      ], criterio: 'romantico' },
    { palabras: [
        'ninos','niños','niñas','bebe','bebé','bebes','bebés',
        'familia','familiar','familias','peques','pequeños',
        'crío','crios','chaval','chavales','infantil',
        'silla de bebe','trona','con el bebe','con el bebé',
        'con los niños','para niños','aptos niños','apto niños',
        'con toda la familia','plan familiar','salir en familia',
      ], criterio: 'ninos' },
    { palabras: [
        'perro','perros','mascota','mascotas','peludo','peludos',
        'admiten perros','con mi perro','traer el perro',
        'venir con perro','perro bienvenido','dog friendly',
      ], criterio: 'mascotas' },
    { palabras: [
        'terraza','terrazas','exterior','exteriores',
        'aire libre','al aire libre','fuera','patio',
        'veladores','parasol','jardin','jardín',
        'comer fuera','sentarse fuera','mesa fuera',
      ], criterio: 'terraza' },
    { palabras: [
        'vistas','vista','panoramica','panorámica',
        'azotea','rooftop','mirador','skyline',
        'vistas a','vistas de','vistas desde',
        'con vistas','buenas vistas',
      ], criterio: 'vistas' },
    { palabras: [
        'musica en directo','música en directo','musica directo',
        'concierto','jazz','flamenco','actuacion','actuación',
        'directo','en vivo','grupo en vivo','banda','cantante',
        'tocan','hay musica','hay música',
      ], criterio: 'musica_directo' },
    { palabras: [
        'tranquilo','tranquila','tranquilos','tranquilas',
        'silencioso','silenciosa','sin ruido','sin música',
        'paz','relajado','relajada','descansar','desconectar',
        'hablar tranquilo','hablar bien','sin jaleo','poco ruido',
        'ambiente tranquilo','sitio tranquilo','lugar tranquilo',
      ], criterio: 'tranquilo' },
    { palabras: [
        'barato','barata','economico','económico','economica','económica',
        'asequible','buen precio','buenos precios','precio razonable',
        'relacion calidad','relación calidad','calidad precio',
        'no muy caro','no caro','precio justo','ajustado',
        'sin gastar mucho','sin gastar demasiado','presupuesto',
        'price','affordable','gasto','gastar poco',
      ], criterio: 'precio_ok' },
    { palabras: [
        'ambiente','bonito','acogedor','acogedora',
        'especial','diferente','original','unico','único','única',
        'curioso','curioso','curioso','con encanto','encantador',
        'chulo','chula','guapo','guapa','precioso','preciosa',
        'decoracion','decoración','decorado','bien decorado',
        'instagram','instagrameable','fotogenico','fotogénico',
        'sorprendente','espectacular','llamativo','llamativa',
        'con personalidad','autentico','auténtico','genuino',
      ], criterio: 'buen_ambiente' },
    { palabras: [
        'marcha','animado','animada','fiesta','fiestas',
        'movida','bullicio','bullicioso','con vida',
        'ambiente animado','mucha gente','lleno','concurrido',
        'noche','nocturno','nocturna','de noche','salir de noche',
        'copas','tomar algo','after','afterwork',
      ], criterio: 'ruidoso' },
    { palabras: [
        'centrico','céntrico','centrica','céntrica',
        'centro','centro madrid','bien ubicado','buena ubicacion',
        'buena ubicación','bien situado','facil llegar','fácil llegar',
        'cerca de todo','en el centro','zona centro',
      ], criterio: 'centrico' },
    { palabras: [
        'mejor','top','recomendado','recomendada','muy valorado',
        'famoso','famosa','conocido','conocida','popular',
        'estrella','estrellas','premiado','premiada',
        'el mejor','lo mejor','numero uno','número uno',
        'mas valorado','más valorado','mejor valorado',
        'excelente reputacion','excelente reputación',
      ], criterio: 'muy_valorado' },
  ];

  if (hasQuery && activeFilters.size === 0) {
    const qn = norm(query);
    let encontrados = [];
    // Añadir espacios al inicio y fin para detectar palabras completas
    const qnPadded = ' ' + qn + ' ';
    for (const intent of INTENCIONES) {
      if (intent.palabras.some(p => {
        // Para keywords cortas (< 5 chars) exigir límite de palabra
        // Para keywords largas, includes() es suficiente
        if (p.length < 5) {
          return qnPadded.includes(' ' + p + ' ');
        }
        return qn.includes(p);
      })) {
        encontrados.push(intent.criterio);
      }
    }
    // Activar todos los criterios detectados (máx 2 para no over-filtrar)
    encontrados.slice(0, 2).forEach(c => activeFilters.add(c));
    if (activeFilters.size > 0) renderFilterChips();
  }

  const locText = hasQuery ? extractLocationText(query) : null;
  const queryPlato = locText
    ? norm(query)
        .replace(/cerca del?\s+[\w\s,áéíóúüñ]+$/i, '')
        .replace(/cerca de (la|los|las)\s+[\w\s,áéíóúüñ]+$/i, '')
        .replace(/en el\s+[\w\s,áéíóúüñ]+$/i, '')
        .replace(/en la\s+[\w\s,áéíóúüñ]+$/i, '')
        .replace(/en [\w\s,áéíóúüñ]+$/i, '')
        .replace(/desde [\w\s,áéíóúüñ]+$/i, '')
        .replace(/por [\w\s,áéíóúüñ]+$/i, '')
        .replace(/junto al?\s+[\w\s,áéíóúüñ]+$/i, '')
        .replace(/al lado del?\s+[\w\s,áéíóúüñ]+$/i, '')
        .replace(/proximo al?\s+[\w\s,áéíóúüñ]+$/i, '')
        .trim()
    : norm(query);
  const stop = new Set([
    'que','con','para','los','las','del','una','uno','por','cerca','desde',
    'muy','mas','más','quiero','busco','dame','restaurante','restaurantes',
    'buenos','buena','buenas','buen','sitio','lugar','ir','un','buen','buena',
    'pareja','familia','ninos','niños','perro','terraza','tranquilo','barato',
    'ambiente','marcha','centrico','mejor','romantico','romántico',
    'animado','vistas','directo',
  ]);
  const qTokens = hasQuery ? tokenize(queryPlato).filter(t => !stop.has(t)) : [];

  // Detectar si la query es una categoría de cocina
  const cocinaBuscada = detectarCocina(queryPlato || query);
  const tieneCocina   = !!cocinaBuscada;
  const tienePlato    = qTokens.length > 0 && !tieneCocina;
  const tieneUbicacion = !!locText;

  // --- Scoring ---
  let scored = [];
  if (tieneCocina) {
    // Búsqueda por categoría de cocina — puntuar por platos representativos
    for (const r of RESTAURANTES) {
      const s = scorePorCocina(r, cocinaBuscada);
      if (s) scored.push(s);
    }
  } else if (tienePlato) {
    for (const r of RESTAURANTES) {
      const s = scoreRestaurante(r, qTokens);
      if (s) scored.push(s);
    }
  } else {
    // No food query, show all (will be filtered by criteria)
    scored = RESTAURANTES.map(r => ({r, ms: 0, mp: []}));
  }

  // --- Apply criteria filters ---
  if (hasFilters) {
    scored = scored.filter(s => [...activeFilters].every(f => pasaCriterio(s.r, f)));
  }

  if (!scored.length) {
  document.getElementById('welcome').style.display = 'none';
  results.style.display = 'flex';
    status.textContent = 'Sin resultados';
    results.innerHTML = '<div class="empty"><div class="e-icon">🔍</div>No hay restaurantes que cumplan todos los criterios. Prueba a quitar algún filtro.</div>';
    return;
  }

  // --- Geocode location ---
  let locCoords = null;
  if (tieneUbicacion) {
    status.innerHTML = '<div class="loading"><div class="dot"></div><div class="dot"></div><div class="dot"></div><span>Localizando "' + locText + '"...</span></div>';
    locCoords = await geocodificarUsuario(locText);
  }

  // --- Sort + filtro de radio máximo ---
  if (locCoords) {
    scored.forEach(s => {
      s.dk = s.r.coords ? haversine(locCoords, s.r.coords) : null;
    });

    // Radio máximo: 2 km si solo hay zona, 4 km si hay plato + zona
    const radioMax = tienePlato ? 4.0 : 2.0;
    const antesRadio = scored.length;
    scored = scored.filter(s => s.dk === null || s.dk <= radioMax);

    // Si el radio deja < 3 resultados, relajarlo automáticamente
    if (scored.length < 3 && antesRadio >= 3) {
      scored = [];
      const radioRelax = radioMax * 2;
      scored = (tienePlato
        ? [...RESTAURANTES].map(r => scoreRestaurante(r, qTokens)).filter(Boolean)
        : RESTAURANTES.map(r => ({r, ms:0, mp:[]}))).filter(s => {
          s.dk = s.r.coords ? haversine(locCoords, s.r.coords) : null;
          return s.dk === null || s.dk <= radioRelax;
        });
      if (hasFilters) scored = scored.filter(s => [...activeFilters].every(f => pasaCriterio(s.r, f)));
    }

    if (tienePlato) {
      // Plato + zona: primero los que tienen match del plato (por score),
      // dentro de cada grupo ordenar por distancia
      scored.sort((a, b) => {
        if (b.ms !== a.ms) return b.ms - a.ms;        // 1. match de plato
        return (a.dk ?? 99) - (b.dk ?? 99);            // 2. distancia
      });
    } else {
      // Solo zona (sin plato): ordenar puramente por distancia,
      // usar pct_positivo como desempate entre restaurantes a igual distancia
      scored.sort((a, b) => {
        const da = a.dk ?? 99, db = b.dk ?? 99;
        if (Math.abs(da - db) > 0.3) return da - db;  // 1. distancia (umbral 300m)
        return b.r.pct_positivo - a.r.pct_positivo;   // 2. calidad como desempate
      });
    }
  } else if (hasFilters && !tienePlato) {
    // Solo criterios sin zona
    if (activeFilters.has('centrico')) {
      // Céntrico: ordenar puramente por distancia al centro
      // Solo desempatar por calidad cuando la diferencia es < 100m
      scored.sort((a,b) => {
        const da = a.r.dist_sol ?? 99, db = b.r.dist_sol ?? 99;
        if (Math.abs(da - db) > 0.1) return da - db;
        return b.r.pct_positivo - a.r.pct_positivo;
      });
    } else {
      // Resto de criterios: ordenar por calidad
      scored.sort((a,b) => b.r.pct_positivo - a.r.pct_positivo || b.r.valoracion_google - a.r.valoracion_google);
    }
  } else {
    // Solo plato sin zona: match primero, calidad como desempate
    scored.sort((a,b) => b.ms !== a.ms ? b.ms - a.ms : b.r.pct_positivo - a.r.pct_positivo);
  }

  const top = scored.slice(0, 10);
  const filterLabel = [...activeFilters].map(k => CRITERIOS[k].label).join(' · ');
  const radioUsado = locCoords ? ((tienePlato || tieneCocina) ? 4.0 : 2.0) : null;
  const cocinaLabel = tieneCocina ? COCINAS[cocinaBuscada].label : '';
  status.textContent = top.length + ' restaurantes' +
    (cocinaLabel ? ' de ' + cocinaLabel : '') +
    (locText ? ' cerca de ' + locText + (radioUsado ? ` (≤${radioUsado} km)` : '') : '') +
    (filterLabel ? ' · ' + filterLabel : '');

  results.innerHTML = top.map((s, i) => {
    const r   = s.r;
    const pct = r.pct_positivo;
    const bc  = pct >= 90 ? 'badge-green' : pct >= 75 ? 'badge-amber' : 'badge-red';
    // Platos: top5 con badge de menciones si es el plato buscado
    const platosData = parseListaPlatos(r.top5_platos);
    const platos = platosData.map(p => {
      const isM = s.mp.some(m => norm(m.nombre).includes(norm(p.nombre)) || norm(p.nombre).includes(norm(m.nombre)));
      const mencionesTag = isM ? ` <span class="plato-menciones">${p.menciones} reseñas</span>` : '';
      return `<span class="plato ${isM ? 'match' : ''}">${p.nombre}${mencionesTag}</span>`;
    }).join('');
  document.getElementById('welcome').style.display = 'none';
  results.style.display = 'flex';
  // Modal click handlers
  setTimeout(() => {
    results.querySelectorAll('.card').forEach((card, idx) => {
      card.addEventListener('click', () => abrirModal(top[idx]));
    });
  }, 0);

    // Servicio destacado — mostrar frase de reseña si existe
    const servicioStr = r.servicio_frases
      ? `<div class="dist" style="color:var(--green);font-style:italic">💬 "${r.servicio_frases.split('|')[0].trim().substring(0,100)}"</div>`
      : '';

    // Razón de relevancia — por qué este restaurante para esta búsqueda
    const razonTexto = generarRazon(r, s, locText, [...activeFilters]);
    const razonStr = razonTexto
      ? `<div class="razon">✦ ${razonTexto}</div>`
      : '';

    // Distancia: mostrar solo cuando aporta info no repetida en la razón
    let distStr = '';
    if (s.dk != null) {
      // Búsqueda por zona — siempre mostrar distancia desde el punto buscado
      distStr = `<div class="dist">📍 ${s.dk.toFixed(2)} km desde ${locText}</div>`;
    }
    // Si el filtro céntrico está activo pero NO hay zona buscada,
    // la razón ya muestra "muy céntrico (X km)" — no duplicar en distStr
    const mapsUrl = 'https://www.google.com/maps/search/' + encodeURIComponent(r.nombre + ' ' + r.direccion + ' Madrid');
    const critBadges = getCriteriaBadgesHtml(r);
    return `<div class="card">
      <div class="card-top">
        <div style="display:flex;gap:10px;min-width:0;flex:1">
          <div class="card-index">${i+1}</div>
          <div class="card-info">
            <div class="card-name">${r.nombre}</div>
            ${r.direccion ? `<div class="card-addr">📍 ${r.direccion}</div>` : ''}
          </div>
        </div>
        <div class="card-right">
          <span class="badge ${bc}">${pct.toFixed(0)}% positivo</span>
          <span class="stars">★ ${r.valoracion_google}</span>
          <a class="maps-link" href="${mapsUrl}" target="_blank">Maps ↗</a>
        </div>
      </div>
      ${critBadges ? `<div class="criteria-badges">${critBadges}</div>` : ''}
      ${razonStr}
      ${platos ? `<div class="platos">${platos}</div>` : ''}
      ${servicioStr}
      ${distStr}
    </div>`;
  }).join('');
}

document.getElementById('query').addEventListener('keydown', e => { if (e.key === 'Enter') buscar(); });

function abrirModal(s) {
  const r = s.r;
  const pct = r.pct_positivo;
  const bc  = pct >= 90 ? '#4caf82' : pct >= 75 ? '#c8963e' : '#c04040';

  const platosList = parseListaPlatos(r.top5_platos);
  const platosHtml = platosList.map(p => `<span style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:3px 10px;font-size:12px;color:#888">${p.nombre}<span style="color:#c8a96e88;font-size:11px;margin-left:4px">(${p.menciones})</span></span>`).join('');

  const badgesHtml = [
    r.score_ambiente >= 7.5 && '<span style="background:#0a1f14;color:#4caf82;border:1px solid #0a2a1a;border-radius:100px;padding:3px 9px;font-size:11px">✨ Buen ambiente</span>',
    r.ruido_neg === 0 && r.score_tranquilo >= 7 && '<span style="background:#0a1818;color:#5ab8b8;border:1px solid #0a2828;border-radius:100px;padding:3px 9px;font-size:11px">🤫 Tranquilo</span>',
    r.score_precio >= 7 && '<span style="background:#0e1a0a;color:#7ab87a;border:1px solid #1a2a1a;border-radius:100px;padding:3px 9px;font-size:11px">💶 Buen precio</span>',
    r.dist_sol <= 1.5 && '<span style="background:#1a1505;color:#c8a96e;border:1px solid #c8a96e22;border-radius:100px;padding:3px 9px;font-size:11px">📍 Céntrico</span>',
    r.flag_terraza && '<span style="background:#1a1208;color:#c8963e;border:1px solid #2a1e08;border-radius:100px;padding:3px 9px;font-size:11px">☀️ Terraza</span>',
    r.flag_romantico && '<span style="background:#1a0a14;color:#c87aaa;border:1px solid #2a1020;border-radius:100px;padding:3px 9px;font-size:11px">🕯️ Romántico</span>',
    r.flag_ninos && '<span style="background:#0e141a;color:#7aaac8;border:1px solid #1a2a3a;border-radius:100px;padding:3px 9px;font-size:11px">👶 Apto niños</span>',
    r.flag_mascotas && '<span style="background:#100e08;color:#a89050;border:1px solid #2a2010;border-radius:100px;padding:3px 9px;font-size:11px">🐾 Mascotas</span>',
    r.flag_vistas && '<span style="background:#081218;color:#5a8aaa;border:1px solid #0a1e2a;border-radius:100px;padding:3px 9px;font-size:11px">🏙️ Con vistas</span>',
    r.flag_musica_directo && '<span style="background:#0e0a1a;color:#8a7ab8;border:1px solid #1a1028;border-radius:100px;padding:3px 9px;font-size:11px">🎵 Música en directo</span>',
  ].filter(Boolean).join('');

  const razonTexto = generarRazon(r, s, null, [...activeFilters]);

  const content = `
    <div class="modal-header">
      <div>
        <div class="modal-title">${r.nombre}</div>
        <div class="modal-addr">📍 ${r.direccion || 'Madrid'}</div>
      </div>
      <button class="modal-close" onclick="document.getElementById('modal-overlay').style.display='none'">×</button>
    </div>

    <div style="display:flex;gap:8px;align-items:center;margin-bottom:14px">
      <span style="background:#0a1f14;color:${bc};border:1px solid #0a2a1a;border-radius:100px;padding:4px 12px;font-size:12px;font-weight:600">${pct}% positivo</span>
      <span style="color:#c8a96e;font-size:13px;font-weight:500">★ ${r.valoracion_google}</span>
      <span style="color:#444;font-size:12px">${r.n_resenas} reseñas</span>
      <a href="https://www.google.com/maps/search/${encodeURIComponent(r.nombre + ' ' + (r.direccion||'') + ' Madrid')}" target="_blank" style="margin-left:auto;color:#5a8aaa;font-size:12px;text-decoration:none">Maps ↗</a>
    </div>

    ${razonTexto ? `<div style="background:#0f0f0f;border-left:2px solid #c8a96e;border-radius:8px;padding:8px 12px;margin-bottom:14px;font-size:12px;color:#888">✦ ${razonTexto}</div>` : ''}

    ${badgesHtml ? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px">${badgesHtml}</div>` : ''}

    ${platosList.length ? `
    <div class="modal-section">
      <div class="modal-section-title">Platos destacados</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px">${platosHtml}</div>
    </div>` : ''}

    ${r.servicio_frases ? `
    <div style="background:#0a1408;border-left:2px solid #4caf82;border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px">
      <div style="font-size:11px;color:#4caf82;font-weight:600;margin-bottom:4px">💬 Servicio</div>
      <div style="font-size:13px;color:#888;font-style:italic">"${r.servicio_frases.split('|')[0].trim().substring(0,120)}"</div>
    </div>` : ''}

    ${r.dist_sol != null ? `<div style="font-size:12px;color:#444;margin-top:4px">📍 ${r.dist_sol} km del centro (Sol)</div>` : ''}
  `;

  document.getElementById('modal-content').innerHTML = content;
  document.getElementById('modal-overlay').style.display = 'flex';
}

function resetear() {
  document.getElementById('query').value = '';
  activeFilters.clear();
  renderFilterChips();
  document.getElementById('results').innerHTML = '';
  document.getElementById('results').style.display = 'none';
  document.getElementById('welcome').style.display = 'flex';
  document.getElementById('status').textContent = '';
  document.getElementById('active-filters').innerHTML = '';
}

function cerrarModal(e) {
  if (e.target.id === 'modal-overlay') {
    document.getElementById('modal-overlay').style.display = 'none';
  }
}

window.addEventListener('DOMContentLoaded', () => {});
"""

N = len(records)

# Build filter chips HTML
filter_chips_html = '\n'.join(
    f'<button class="filter-chip" id="fc-{key}" onclick="toggleFilter(\'{key}\')">'
    f'<span class="icon">{data["icon"]}</span>{data["label"]}</button>'
    for key, data in {
        'buen_ambiente':  {'label': 'Buen ambiente',     'icon': '✨'},
        'tranquilo':      {'label': 'Tranquilo',          'icon': '🤫'},
        'ruidoso':        {'label': 'Con marcha',         'icon': '🎉'},
        'precio_ok':      {'label': 'Buen precio',        'icon': '💶'},
        'centrico':       {'label': 'Céntrico',            'icon': '📍'},
        'muy_valorado':   {'label': 'Muy valorado',       'icon': '⭐'},
        'terraza':        {'label': 'Terraza',            'icon': '☀️'},
        'vistas':         {'label': 'Con vistas',         'icon': '🏙️'},
        'ninos':          {'label': 'Apto niños',         'icon': '👶'},
        'mascotas':       {'label': 'Admite mascotas',    'icon': '🐾'},
        'musica_directo': {'label': 'Música en directo',  'icon': '🎵'},
        'romantico':      {'label': 'Romántico',          'icon': '🕯️'},
    }.items()
)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Restaurantes Madrid</title>
<style>{CSS}</style>
</head>
<body>
<div class="app">

  <div class="header">
    <div class="header-inner">
      <div class="logo-wrap">🍽️</div>
      <div>
        <div style="font-family:'DM Serif Display',serif;font-size:18px;color:#f0ece4;letter-spacing:-0.3px">Restaurantes <span style="color:#c8a96e">Madrid</span></div>
        <div class="sub">{N} restaurantes · análisis de reseñas</div>
      </div>
    </div>
    <button class="btn-nuevo" onclick="resetear()">Nueva búsqueda</button>
  </div>

  <div class="results-area" id="results-area">
    <div class="welcome" id="welcome">
      <div class="welcome-icon">🍽️</div>
      <h2>¿Qué te apetece hoy?</h2>
      <p>Busca por plato, zona, cocina o cuéntame qué ocasión es</p>
      <div class="sugs">
<button class="sug" onclick="setQ('Quiero un restaurante indio')">Quiero un restaurante indio</button>
<button class="sug" onclick="setQ('Cena romántica para dos')">Cena romántica para dos</button>
<button class="sug" onclick="setQ('¿Dónde puedo comer cachopo?')">¿Dónde puedo comer cachopo?</button>
<button class="sug" onclick="setQ('Apto para niños')">Apto para niños</button>
<button class="sug" onclick="setQ('Mejor relación calidad-precio')">Mejor relación calidad-precio</button>
<button class="sug" onclick="setQ('Algo con terraza')">Algo con terraza</button>
<button class="sug" onclick="setQ('Cerca de Malasaña')">Cerca de Malasaña</button>
      </div>
    </div>
    <div id="results" style="display:none;flex-direction:column;gap:10px"></div>
  </div>

  <div class="footer">
    <div class="filters-section">
      <div class="filters-title">Filtrar por criterio</div>
      <div class="filters-grid">
<button class="filter-chip" id="fc-buen_ambiente" onclick="toggleFilter('buen_ambiente')"><span class="icon">✨</span>Buen ambiente</button>
<button class="filter-chip" id="fc-tranquilo" onclick="toggleFilter('tranquilo')"><span class="icon">🤫</span>Tranquilo</button>
<button class="filter-chip" id="fc-ruidoso" onclick="toggleFilter('ruidoso')"><span class="icon">🎉</span>Con marcha</button>
<button class="filter-chip" id="fc-precio_ok" onclick="toggleFilter('precio_ok')"><span class="icon">💶</span>Buen precio</button>
<button class="filter-chip" id="fc-centrico" onclick="toggleFilter('centrico')"><span class="icon">📍</span>Céntrico</button>
<button class="filter-chip" id="fc-muy_valorado" onclick="toggleFilter('muy_valorado')"><span class="icon">⭐</span>Muy valorado</button>
<button class="filter-chip" id="fc-terraza" onclick="toggleFilter('terraza')"><span class="icon">☀️</span>Terraza</button>
<button class="filter-chip" id="fc-vistas" onclick="toggleFilter('vistas')"><span class="icon">🏙️</span>Con vistas</button>
<button class="filter-chip" id="fc-ninos" onclick="toggleFilter('ninos')"><span class="icon">👶</span>Apto niños</button>
<button class="filter-chip" id="fc-mascotas" onclick="toggleFilter('mascotas')"><span class="icon">🐾</span>Mascotas</button>
<button class="filter-chip" id="fc-musica_directo" onclick="toggleFilter('musica_directo')"><span class="icon">🎵</span>Música en directo</button>
<button class="filter-chip" id="fc-romantico" onclick="toggleFilter('romantico')"><span class="icon">🕯️</span>Romántico</button>
      </div>
    </div>
    <div class="status-row">
      <div class="status" id="status"></div>
      <div class="active-filters" id="active-filters"></div>
    </div>
    <div class="input-wrap">
      <input id="query" type="text" placeholder="Plato, zona, cocina o situación..." />
      <button class="btn-search" onclick="buscar()">↑</button>
    </div>
    <div class="hint">↵ Enter para buscar</div>
  </div>

</div>

<div class="modal-overlay" id="modal-overlay" style="display:none" onclick="cerrarModal(event)">
  <div class="modal" id="modal">
    <div class="modal-handle"></div>
    <div id="modal-content"></div>
  </div>
</div>

<script>
{DATA_JS}
{JS}
</script>
</body>
</html>"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✔ Generado: {OUTPUT_HTML}")
print(f"  Servidor: python3 -m http.server 8080")
print(f"  URL:      http://localhost:8080/agente_restaurantes.html")
