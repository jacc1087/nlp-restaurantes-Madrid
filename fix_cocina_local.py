import re, math, unicodedata, pandas as pd

def _n(s):
    s = s.lower().strip()
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')

# ── Capa 1: por nombre ────────────────────────────────────────────────────────
NOMBRE_KEYWORDS = {
    'gallega':    ['galicia', 'gallego', 'gallega', 'montes de galicia', 'morgana'],
    'asturiana':  ['asturian', 'asturias', 'eiffel restaurante'],
    'vasca':      ['txirimiri', 'dantxari', 'txoko', 'euskal', 'juana la loca'],
    'andaluza':   ['gaditana', 'gaditano', 'sevill', 'andaluz'],
    'madrileña':  ['madril', 'casa toni', 'casa lucas', 'casa carola', 'angelita madrid',
                   'gastro', 'taberna la concha', 'la castela', 'meson molinero',
                   'taberna malaspina', 'el rincon del cabo', 'vinos de bellota',
                   'la descarada', 'salon arte', 'sazon', 'cafe astral'],
    'argentina':  ['argentin', 'pampa beef', 'cabaña argentina', 'bayres', 'asado central',
                   'camoati', 'el social', 'mu! el placer', 'mu el placer'],
    'italiana':   ['trattoria', 'pizzeria', 'pizzart', 'mozzarell', 'napoli', 'piamonte',
                   'fusco', 'pastamore', 'malafemmena', 'malatesta', 'pulcinella',
                   'maruzzella', 'oliveto', 'piccola', 'davanti', 'bresca'],
    'japonesa':   ['sibuya', 'hotaru', 'miyama', 'ichikani', 'dokidoki', 'kaiten sushi',
                   'sr.ito', 'sakale'],
    'india':      ['indian', 'tandoori', 'bangalore', 'kathmandu', 'purnima',
                   'radhuni', 'curry masala', 'indian aroma'],
    'mexicana':   ['taco bar', 'mawey', 'el rey de los tacos', 'tacos &'],
    'peruana':    ['kausa', 'quispe', 'tampu', 'ronda 14'],
    'venezolana': ['grama lounge'],
    'arabe':      ['hummuseria', 'beytna'],
    'americana':  ['steakburger', 'steak burger', 'hamburgues', 'burnout', 'brew wild'],
    'fusion':     ['diverxo', 'streetxo', 'dstage', 'bacira', 'coque', 'bestial',
                   'casa jaguar'],
    'arroceria':  ['arroceria', 'arrocería'],
    'tailandesa': ['bambúbox', 'bambubox'],
}

# ── Capa 2: por platos exclusivos ─────────────────────────────────────────────
PLATOS_COCINA = {
    'gallega':    ['empanada gallega', 'percebes', 'vieiras', 'caldo gallego', 'lacon',
                   'filloas', 'pote gallego', 'zorza', 'cocido gallego'],
    'asturiana':  ['cachopo', 'cachopu', 'fabada asturiana', 'fabada', 'oricios',
                   'cabrales', 'tortos', 'frixuelos', 'verdinas'],
    'vasca':      ['pintxos', 'pintxo', 'gilda', 'kokotxas', 'cocochas', 'txangurro',
                   'marmitako', 'txakoli', 'pil pil', 'angulas'],
    'andaluza':   ['pescaito frito', 'tortillitas de camarones', 'ajoblanco',
                   'berenjenas con miel', 'espinacas con garbanzos', 'cola de toro'],
    'madrileña':  ['cocido madrileño', 'cocido madrileno', 'callos a la madrileña',
                   'bocadillo de calamares', 'soldaditos de pavia'],
    'italiana':   ['carbonara', 'cacio e pepe', 'amatriciana', 'ossobuco', 'panna cotta',
                   'tiramisú', 'tiramisu', 'tagliatelle', 'pappardelle', 'gnocchi',
                   'cannoli', 'arancini', 'burrata', 'lasaña', 'lasana', 'pizza',
                   'bruschetta', 'focaccia'],
    'peruana':    ['lomo saltado', 'lomo salteado', 'causa', 'causa limeña', 'anticuchos',
                   'aji de gallina', 'leche de tigre', 'chaufa', 'tiradito',
                   'ceviche amazónico', 'ceviche pachamanquero'],
    'japonesa':   ['ramen', 'sushi', 'sashimi', 'nigiri', 'gyozas', 'gyoza', 'tempura',
                   'udon', 'mochi', 'edamame', 'yakitori', 'takoyaki'],
    'india':      ['tikka masala', 'biryani', 'naan', 'samosa', 'korma', 'dal',
                   'tandoori', 'butter chicken', 'palak paneer'],
    'mexicana':   ['tacos', 'burrito', 'quesadilla', 'fajitas', 'enchilada', 'pozole',
                   'carnitas', 'chilaquiles', 'tamales'],
    'venezolana': ['arepa', 'arepas', 'pabellon', 'cachapa', 'hallaca', 'tequeños',
                   'mandocas', 'caraotas'],
    'argentina':  ['entraña', 'mollejas', 'chorizo criollo', 'empanadas argentinas',
                   'empanadas criollas', 'lomo alto', 'chimichurri', 'provoleta',
                   'filete argentino', 'entraña vacio'],
    'arabe':      ['shawarma', 'falafel', 'tabule', 'baba ganoush', 'labneh',
                   'shakshuka', 'kibbeh', 'couscous'],
    'americana':  ['smash burger', 'pulled pork', 'costillas bbq', 'mac and cheese',
                   'chicken wings', 'brisket'],
    'griega':     ['gyros', 'souvlaki', 'moussaka', 'spanakopita', 'baklava', 'dolmades'],
    'china':      ['dim sum', 'wonton', 'pato pekin', 'chow mein', 'dumplings'],
    'tailandesa': ['pad thai', 'tom yum', 'massaman', 'satay', 'curry panang'],
    'francesa':   ['foie gras', 'confit de pato', 'magret', 'bouillabaisse', 'escargots',
                   'cassoulet'],
    'colombiana': ['bandeja paisa', 'ajiaco', 'sancocho'],
}

def detectar(nombre, todos_platos):
    nombre_n = _n(nombre)
    # Capa 1: nombre
    for cocina, kws in NOMBRE_KEYWORDS.items():
        if any(kw in nombre_n for kw in kws):
            return cocina
    # Capa 2: platos exclusivos
    if not todos_platos or str(todos_platos).strip() in ('', 'nan'):
        return ''
    platos = {}
    for p in str(todos_platos).split(','):
        m = re.match(r'^(.+?)\((\d+)\)$', p.strip())
        if m:
            platos[_n(m.group(1).strip())] = int(m.group(2))
    if not platos:
        return ''
    total = sum(platos.values())
    mejor, mejor_score = '', 0.0
    for cocina, defs in PLATOS_COCINA.items():
        matches, menciones = [], 0
        for d in defs:
            dn = _n(d)
            for pr, mn in platos.items():
                if dn == pr or (len(dn) > 5 and dn in pr):
                    if mn >= 2:
                        matches.append(mn)
                        menciones += mn
                    break
        if len(matches) < 2:
            if cocina == 'asturiana' and any(m >= 5 for m in matches):
                pass
            else:
                continue
        if menciones / total < 0.25:
            continue
        score = sum(2 + math.log2(m) for m in matches)
        if score > mejor_score:
            mejor_score, mejor = score, cocina
    return mejor

df = pd.read_csv('analisis_restaurantes.csv')
df['cocina_detectada'] = df.apply(
    lambda r: detectar(str(r['nombre']), str(r['todos_platos'])), axis=1
)

from collections import Counter, defaultdict
print('DISTRIBUCIÓN:')
print(df['cocina_detectada'].replace('', '(sin categoría)').value_counts().to_string())
print()
por_cocina = defaultdict(list)
for _, r in df.iterrows():
    por_cocina[r['cocina_detectada']].append(r['nombre'])
for cocina in sorted(por_cocina):
    print(f'[{cocina or "SIN CATEGORÍA"}]')
    for n in por_cocina[cocina]: print(f'  {n}')
    print()

df.to_csv('analisis_restaurantes.csv', index=False)
print('Guardado.')
