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
cols_opcionales = ['todos_platos'] + [f'criterio_{c}' for c in
    ['ninos','mascotas','terraza','vistas','musica_directo','romantico']]
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
for c in ['todos_platos'] + [f'criterio_{x}' for x in ['ninos','mascotas','terraza','vistas','musica_directo','romantico']]:
    if c in df2.columns:
        df2[c] = df2[c].fillna('' if 'platos' in c else False)
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
  --bg: #f7f5f0;
  --surface: #ffffff;
  --surface2: #f2efe8;
  --border: #e8e3d8;
  --text: #1c1a16;
  --text2: #6b6560;
  --text3: #9b958e;
  --accent: #c8702a;
  --accent-bg: #fdf3eb;
  --green: #2d7a4f;
  --green-bg: #e8f5ee;
  --amber: #a06020;
  --amber-bg: #fef3e2;
  --red: #b03030;
  --red-bg: #fde8e8;
  --blue: #2563eb;
  --blue-bg: #dbeafe;
  --tag-bg: #ede9e0;
  --shadow: 0 1px 4px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
  --radius: 14px;
}

body {
  font-family: 'DM Sans', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 2.5rem 1rem 4rem;
}

.container { max-width: 780px; margin: 0 auto; }

.header { margin-bottom: 2rem; }
.header h1 {
  font-family: 'DM Serif Display', Georgia, serif;
  font-size: clamp(24px, 4vw, 32px);
  font-weight: 400;
  letter-spacing: -0.02em;
  color: var(--text);
  line-height: 1.2;
}
.header h1 span { color: var(--accent); }
.header .sub {
  font-size: 13px;
  color: var(--text2);
  margin-top: 4px;
  font-weight: 300;
}

/* ── Search bar ── */
.search-row {
  display: flex;
  gap: 8px;
  margin-bottom: 1.25rem;
}
.search-wrap {
  flex: 1;
  position: relative;
}
.search-wrap svg {
  position: absolute;
  left: 14px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text3);
  pointer-events: none;
}
#query {
  width: 100%;
  font-family: inherit;
  font-size: 14px;
  padding: 0 14px 0 40px;
  height: 46px;
  border: 1.5px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  outline: none;
  color: var(--text);
  transition: border-color .15s;
}
#query:focus { border-color: var(--accent); }
#query::placeholder { color: var(--text3); }

.btn-search {
  height: 46px;
  padding: 0 22px;
  font-family: inherit;
  font-size: 13px;
  font-weight: 600;
  border: none;
  border-radius: 12px;
  background: var(--text);
  color: #fff;
  cursor: pointer;
  transition: background .15s, transform .1s;
  white-space: nowrap;
}
.btn-search:hover { background: #333; }
.btn-search:active { transform: scale(.97); }

/* ── Criteria filters ── */
.filters-section {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  margin-bottom: 1.25rem;
}
.filters-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--text2);
  margin-bottom: .75rem;
  display: flex;
  align-items: center;
  gap: 6px;
}
.filters-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}
.filter-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12.5px;
  font-weight: 500;
  padding: 7px 14px;
  border: 1.5px solid var(--border);
  border-radius: 100px;
  cursor: pointer;
  background: var(--surface);
  color: var(--text2);
  transition: all .15s;
  user-select: none;
  white-space: nowrap;
}
.filter-chip:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-bg);
}
.filter-chip.active {
  border-color: var(--accent);
  background: var(--accent);
  color: #fff;
}
.filter-chip .icon { font-size: 14px; line-height: 1; }

/* ── Quick chips ── */
.quick-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--text3);
  margin-bottom: .5rem;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 1.25rem;
}
.chip {
  font-size: 12px;
  padding: 6px 13px;
  border: 1.5px solid var(--border);
  border-radius: 100px;
  cursor: pointer;
  color: var(--text2);
  background: var(--surface);
  transition: all .12s;
  font-weight: 400;
}
.chip:hover {
  border-color: var(--text);
  color: var(--text);
  background: var(--surface2);
}

/* ── Status & active filters display ── */
.status-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: .85rem;
  min-height: 24px;
  flex-wrap: wrap;
  gap: 6px;
}
.status { font-size: 13px; color: var(--text2); }
.active-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.af-badge {
  font-size: 11px;
  padding: 3px 10px;
  background: var(--accent-bg);
  color: var(--accent);
  border-radius: 100px;
  font-weight: 500;
  border: 1px solid rgba(200,112,42,.25);
}

/* ── Results ── */
.results { display: flex; flex-direction: column; gap: 10px; }

.card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.25rem;
  box-shadow: var(--shadow);
  transition: box-shadow .15s, border-color .15s;
}
.card:hover { border-color: #d4cfc6; box-shadow: 0 2px 8px rgba(0,0,0,.10), 0 8px 24px rgba(0,0,0,.07); }

.card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.card-index {
  font-family: 'DM Serif Display', Georgia, serif;
  font-size: 22px;
  color: var(--border);
  line-height: 1;
  flex-shrink: 0;
  margin-top: 2px;
}
.card-info { flex: 1; min-width: 0; }
.card-name {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.card-addr {
  font-size: 12px;
  color: var(--text3);
  margin-top: 3px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.card-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 5px;
  flex-shrink: 0;
}
.badge {
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 100px;
  font-weight: 600;
  letter-spacing: .01em;
}
.badge-green { background: var(--green-bg); color: var(--green); }
.badge-amber { background: var(--amber-bg); color: var(--amber); }
.badge-red   { background: var(--red-bg); color: var(--red); }
.stars { font-size: 12px; color: var(--text2); font-weight: 500; }
.maps-link {
  font-size: 11px;
  color: var(--blue);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 3px;
}
.maps-link:hover { text-decoration: underline; }

/* ── Criteria badges on card ── */
.criteria-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 8px;
}
.crit-badge {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 100px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 3px;
}
.crit-badge.good   { background: var(--green-bg); color: var(--green); }
.crit-badge.warn   { background: var(--amber-bg); color: var(--amber); }
.crit-badge.info   { background: var(--blue-bg); color: var(--blue); }
.crit-badge.neutral { background: var(--tag-bg); color: var(--text2); }

/* ── Platos ── */
.platos {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 6px;
}
.plato {
  font-size: 11px;
  padding: 3px 9px;
  background: var(--tag-bg);
  border-radius: 100px;
  color: var(--text2);
}
.plato.match {
  background: var(--blue-bg);
  color: var(--blue);
  font-weight: 600;
}
.plato-menciones {
  font-size: 10px;
  font-weight: 700;
  background: var(--blue);
  color: #fff;
  padding: 1px 5px;
  border-radius: 8px;
  margin-left: 3px;
  vertical-align: middle;
}

.dist { font-size: 12px; color: var(--text2); margin-top: 7px; display: flex; align-items: center; gap: 4px; }

.empty {
  text-align: center;
  padding: 3rem 2rem;
  color: var(--text3);
  font-size: 14px;
  background: var(--surface);
  border: 1.5px dashed var(--border);
  border-radius: var(--radius);
}
.empty .e-icon { font-size: 32px; margin-bottom: .5rem; }

.loading {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text2);
  font-size: 13px;
  padding: .5rem 0;
}
.dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes pulse { 0%,80%,100%{opacity:.2} 40%{opacity:1} }

.sep { width: 100%; height: 1px; background: var(--border); margin: .4rem 0; }

@media(max-width:480px){
  body { padding: 1.5rem .75rem 3rem; }
  .search-row { flex-direction: column; }
  .btn-search { width: 100%; justify-content: center; }
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
async function geocodificarUsuario(lugar) {
  const key = lugar.toLowerCase();
  if (_geoCache[key]) return _geoCache[key];
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
  } catch(e) { console.error('Nominatim error:', e); }
  return null;
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
        break;
      }
    }
  }
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
      return r.dist_sol !== null && r.dist_sol <= 2.5;
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

  if (r.dist_sol !== null && r.dist_sol <= 1.5) badges.push({cls:'info', icon:'📍', text:'Muy céntrico'});
  else if (r.dist_sol !== null && r.dist_sol <= 2.5) badges.push({cls:'neutral', icon:'📍', text:'Céntrico'});

  if (r.pct_positivo >= 95) badges.push({cls:'good', icon:'⭐', text:'Top reseñas'});

  if (r.flag_terraza)         badges.push({cls:'info',    icon:'☀️',  text:'Terraza'});
  if (r.flag_vistas)          badges.push({cls:'info',    icon:'🏙️', text:'Con vistas'});
  if (r.flag_ninos)    badges.push({cls:'neutral', icon:'👶', text:'Apto niños'});
  if (r.flag_mascotas) badges.push({cls:'neutral', icon:'🐾', text:'Mascotas'});
  if (r.flag_musica_directo)  badges.push({cls:'neutral', icon:'🎵', text:'Música en directo'});
  if (r.flag_romantico)       badges.push({cls:'neutral', icon:'🕯️', text:'Romántico'});

  return badges.slice(0, 5).map(b =>
    `<span class="crit-badge ${b.cls}">${b.icon} ${b.text}</span>`
  ).join('');
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

function setQ(q) { document.getElementById('query').value = q; buscar(); }

// ─── Main search ─────────────────────────────────────────────────
async function buscar() {
  const query   = document.getElementById('query').value.trim();
  const status  = document.getElementById('status');
  const results = document.getElementById('results');
  const afRow   = document.getElementById('active-filters');
  results.innerHTML = '';

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
  const stop = new Set(['que','con','para','los','las','del','una','uno','por','cerca','desde','muy','mas','más','quiero','busco','dame','restaurante','restaurantes','buenos','buena','buenas']);
  const qTokens = hasQuery ? tokenize(queryPlato).filter(t => !stop.has(t)) : [];

  const tienePlato     = qTokens.length > 0;
  const tieneUbicacion = !!locText;

  // --- Scoring ---
  let scored = [];
  if (tienePlato) {
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
      scored = scored.concat(
        [...Array.from({length: antesRadio}, (_,i) => i)]
          .map(i => ({ ...scored[i] }))  // no-op, just re-run without filter
      );
      // Re-filter with relaxed radius
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

    scored.sort((a,b) => b.ms !== a.ms ? b.ms - a.ms : (a.dk ?? 99) - (b.dk ?? 99));
  } else if (hasFilters && !tienePlato) {
    scored.sort((a,b) => b.r.pct_positivo - a.r.pct_positivo || b.r.valoracion_google - a.r.valoracion_google);
  } else {
    scored.sort((a,b) => b.ms !== a.ms ? b.ms - a.ms : b.r.pct_positivo - a.r.pct_positivo);
  }

  const top = scored.slice(0, 10);
  const filterLabel = [...activeFilters].map(k => CRITERIOS[k].label).join(' · ');
  const radioUsado = locCoords ? (tienePlato ? 4.0 : 2.0) : null;
  status.textContent = top.length + ' restaurantes' +
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

    // Distancia: mostrar siempre que tengamos coords del restaurante
    let distStr = '';
    if (s.dk != null) {
      // Búsqueda por zona — mostrar distancia desde el punto buscado
      distStr = `<div class="dist">📍 ${s.dk.toFixed(2)} km desde ${locText}</div>`;
    } else if (!locText && r.coords && r.dist_sol != null) {
      // Sin zona buscada — mostrar distancia al centro solo si filtro céntrico activo
      if (activeFilters.has('centrico')) {
        distStr = `<div class="dist">📍 ${r.dist_sol} km del centro</div>`;
      }
    }
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
      ${platos ? `<div class="platos">${platos}</div>` : ''}
      ${distStr}
    </div>`;
  }).join('');
}

document.getElementById('query').addEventListener('keydown', e => { if (e.key === 'Enter') buscar(); });

// Init: show top-rated on load
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('status').textContent = 'Selecciona criterios o busca un plato';
});
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
<div class="container">

  <div class="header">
    <h1>Restaurantes <span>Madrid</span></h1>
    <p class="sub">{N} restaurantes · análisis de reseñas · busca por plato, zona o criterio</p>
  </div>

  <div class="search-row">
    <div class="search-wrap">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input id="query" type="text" placeholder="Plato o zona · ej: croquetas cerca de Sol, sushi en Malasaña..." />
    </div>
    <button class="btn-search" onclick="buscar()">Buscar</button>
  </div>

  <div class="filters-section">
    <div class="filters-title">
      <span>🎛</span> Filtrar por criterio
    </div>
    <div class="filters-grid">
      {filter_chips_html}
    </div>
  </div>

  <div class="quick-label">Búsquedas rápidas</div>
  <div class="chips">
    <span class="chip" onclick="setQ('croquetas de jamón')">croquetas de jamón</span>
    <span class="chip" onclick="setQ('tikka masala')">tikka masala</span>
    <span class="chip" onclick="setQ('rabo de toro')">rabo de toro</span>
    <span class="chip" onclick="setQ('paella cerca de Retiro')">paella cerca de Retiro</span>
    <span class="chip" onclick="setQ('hamburguesa cerca de Gran Vía')">hamburguesa cerca de Gran Vía</span>
    <span class="chip" onclick="setQ('cocido madrileño')">cocido madrileño</span>
    <span class="chip" onclick="setQ('sushi en Malasaña')">sushi en Malasaña</span>
    <span class="chip" onclick="setQ('tarta de queso')">tarta de queso</span>
  </div>

  <div class="status-row">
    <div class="status" id="status"></div>
    <div class="active-filters" id="active-filters"></div>
  </div>

  <div id="results" class="results"></div>
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
