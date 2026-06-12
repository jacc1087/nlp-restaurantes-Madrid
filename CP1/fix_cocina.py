import re, math, unicodedata, pandas as pd
from collections import defaultdict

def _n(s):
    s = s.lower().strip()
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')

# ── Capa 1: por nombre ────────────────────────────────────────────────────────
# Cada entrada verificada contra platos reales del CSV
NOMBRE_KEYWORDS = {
    'gallega':     ['galicia', 'gallego', 'gallega', 'montes de galicia', 'morgana'],
    'asturiana':   ['asturian', 'asturias', 'eiffel restaurante'],
    'vasca':       ['txirimiri', 'dantxari', 'txoko', 'euskal', 'juana la loca'],
    'madrileña':   ['casa toni', 'casa lucas', 'casa carola', 'amordmadre',
                    'la descarada', 'lhardy', 'sobrino de botin', 'botin'],
    'argentina':   ['argentin', 'pampa beef', 'cabaña argentina', 'bayres',
                    'asado central', 'camoati', 'el social',
                    'mu! el placer', 'mu el placer'],
    'italiana':    ['trattoria', 'pizzeria', 'pizzart', 'mozzarell', 'napoli',
                    'piamonte', 'fusco', 'pastamore', 'malafemmena', 'malatesta',
                    'pulcinella', 'maruzzella', 'oliveto', 'piccola', 'davanti',
                    'bresca'],
    'japonesa':    ['sibuya', 'hotaru', 'miyama', 'ichikani', 'dokidoki',
                    'kaiten sushi', 'sr.ito', 'sakale'],
    'india':       ['indian', 'tandoori', 'bangalore', 'kathmandu', 'purnima',
                    'radhuni', 'curry masala', 'indian aroma', 'arpit'],
    'mexicana':    ['taco bar', 'mawey', 'el rey de los tacos', 'tacos &',
                    'la encomienda'],
    'peruana':     ['kausa', 'quispe', 'tampu', 'ronda 14'],
    'venezolana':  ['grama lounge'],
    'arabe':       ['hummuseria', 'beytna'],
    'americana':   ['steakburger', 'steak burger', 'hamburgues', 'burnout'],
    'fusion':      ['diverxo', 'streetxo', 'dstage', 'bacira', 'coque',
                    'bestial', 'casa jaguar'],
    'tailandesa':  ['bambúbox', 'bambubox'],
    # Categorías especiales verificadas por platos
    'arroceria':   ['arroceria', 'arrocería', 'casa benigna',
                    'taberna de peñalver', 'taberna de penalver',
                    'taberna el arco'],
    'asador':      ['meson restaurante la mi venta', 'meson molinero',
                    'camino sacramento'],
    'marisqueria': ['alabaster'],
    # Excluidos explícitamente de italiana (tiramisú anecdótico, carta española)
    'sin_categoria': ['la burbujería', 'la burbujeria'],
}

# ── Capa 2: por platos exclusivos ─────────────────────────────────────────────
PLATOS_COCINA = {
    'gallega':    ['empanada gallega', 'percebes', 'vieiras', 'caldo gallego',
                   'lacon', 'filloas', 'pote gallego', 'zorza', 'cocido gallego'],
    'asturiana':  ['cachopo', 'cachopu', 'fabada asturiana', 'fabada', 'oricios',
                   'cabrales', 'tortos', 'frixuelos', 'verdinas'],
    'vasca':      ['pintxos', 'pintxo', 'gilda', 'kokotxas', 'cocochas',
                   'txangurro', 'marmitako', 'txakoli', 'pil pil', 'angulas'],
    'andaluza':   ['pescaito frito', 'tortillitas de camarones', 'ajoblanco',
                   'berenjenas con miel', 'espinacas con garbanzos', 'cola de toro'],
    'madrileña':  ['cocido madrileño', 'cocido madrileno', 'callos a la madrileña',
                   'callos madrilenos', 'bocadillo de calamares', 'soldaditos de pavia',
                   'callos', 'cocido', 'oreja', 'oreja a la plancha', 'gallinejas',
                   'entresijos', 'caracoles', 'sesos', 'manitas de cerdo', 'manitas',
                   'lengua estofada', 'perdiz estofada', 'potaje', 'judias con chorizo'],
    'italiana':   ['carbonara', 'cacio e pepe', 'amatriciana', 'ossobuco',
                   'panna cotta', 'tiramisú', 'tiramisu', 'tagliatelle', 'pappardelle',
                   'gnocchi', 'cannoli', 'arancini', 'burrata', 'lasaña', 'lasana',
                   'pizza', 'bruschetta', 'focaccia'],
    'peruana':    ['lomo saltado', 'lomo salteado', 'causa', 'causa limeña',
                   'anticuchos', 'aji de gallina', 'leche de tigre', 'chaufa',
                   'tiradito', 'ceviche amazónico', 'ceviche pachamanquero'],
    'japonesa':   ['ramen', 'sushi', 'sashimi', 'nigiri', 'gyozas', 'gyoza',
                   'tempura', 'udon', 'mochi', 'edamame', 'yakitori', 'takoyaki'],
    'india':      ['tikka masala', 'biryani', 'naan', 'samosa', 'korma', 'dal',
                   'tandoori', 'butter chicken', 'palak paneer'],
    'mexicana':   ['tacos', 'burrito', 'quesadilla', 'fajitas', 'enchilada',
                   'pozole', 'carnitas', 'chilaquiles', 'tamales'],
    'venezolana': ['arepa', 'arepas', 'pabellon', 'cachapa', 'hallaca',
                   'tequeños', 'mandocas', 'caraotas'],
    'argentina':  ['entraña', 'mollejas', 'chorizo criollo', 'empanadas argentinas',
                   'empanadas criollas', 'lomo alto', 'chimichurri', 'provoleta'],
    'arabe':      ['shawarma', 'falafel', 'tabule', 'baba ganoush', 'labneh',
                   'shakshuka', 'kibbeh', 'couscous'],
    'americana':  ['smash burger', 'pulled pork', 'costillas bbq', 'mac and cheese',
                   'chicken wings', 'brisket'],
    'griega':     ['gyros', 'souvlaki', 'moussaka', 'spanakopita', 'baklava', 'dolmades'],
    'china':      ['dim sum', 'wonton', 'pato pekin', 'chow mein', 'dumplings'],
    'tailandesa': ['pad thai', 'tom yum', 'massaman', 'satay', 'curry panang'],
    'francesa':   ['foie gras', 'confit de pato', 'magret', 'bouillabaisse',
                   'escargots', 'cassoulet'],
    'colombiana': ['bandeja paisa', 'ajiaco', 'sancocho'],
}

# ── Platos para categoria_carta ───────────────────────────────────────────────
PLATOS_CATEGORIA = {
    'asador':      ['chuleton', 'chuletón', 'lechazo', 'entraña', 'lomo alto',
                    'costillar', 'secreto iberico', 'secreto ibérico', 'mollejas',
                    'cordero asado'],
    'marisqueria': ['percebes', 'vieiras', 'navajas', 'ostras', 'cigalas',
                    'bogavante', 'langosta', 'carabineros', 'centollo', 'buey de mar'],
    'arroceria':   ['paella valenciana', 'arroz negro', 'arroz con bogavante',
                    'arroz señoret', 'fideuá', 'fideua', 'paella de marisco',
                    'paella mixta', 'arroz meloso', 'arroz abanda'],
    'japones':     ['sushi', 'ramen', 'sashimi', 'gyozas', 'gyoza', 'tempura',
                    'udon', 'mochi', 'edamame', 'yakitori', 'nigiri'],
    'italiano':    ['carbonara', 'tiramisú', 'tiramisu', 'tagliatelle', 'pappardelle',
                    'gnocchi', 'lasaña', 'lasana', 'pizza', 'bruschetta', 'focaccia',
                    'arancini', 'burrata', 'panna cotta'],
    'indio':       ['tikka masala', 'biryani', 'naan', 'samosa', 'korma',
                    'tandoori', 'butter chicken', 'palak paneer'],
    'mexicano':    ['tacos', 'quesadilla', 'burrito', 'fajitas', 'enchilada',
                    'pozole', 'carnitas'],
    'peruano':     ['lomo saltado', 'lomo salteado', 'causa', 'tiradito',
                    'anticuchos', 'aji de gallina', 'leche de tigre', 'chaufa',
                    'ceviche amazónico', 'ceviche pachamanquero'],
    'venezolano':  ['arepa', 'arepas', 'pabellon', 'cachapa', 'tequeños', 'hallaca'],
    'argentino':   ['entraña', 'mollejas', 'chorizo criollo', 'empanadas argentinas',
                    'empanadas criollas', 'lomo alto', 'chimichurri', 'provoleta'],
    'tailandes':   ['pad thai', 'tom yum', 'massaman', 'curry panang', 'satay'],
    'arabe':       ['shawarma', 'falafel', 'hummus', 'tabule', 'shakshuka',
                    'baba ganoush', 'labneh', 'couscous'],
    'hamburgueseria': ['smash burger', 'pulled pork', 'costillas bbq',
                       'mac and cheese', 'chicken wings', 'brisket'],
}

# Restaurantes que deben quedar sin categoría aunque matcheen por platos
# (carta española mixta con algún plato internacional anecdótico)
EXCLUIR_CATEGORIA = {
    'la burbujeria',   # tiramisú anecdótico, carta española
}

MIN_PLATOS_MARISQUERIA = 3
MIN_PLATOS     = 2
MIN_MENCIONES  = 3
MIN_PROPORCION = 0.20

def _parsear_platos(todos_platos):
    platos = {}
    for p in str(todos_platos).split(','):
        m = re.match(r'^(.+?)\((\d+)\)$', p.strip())
        if m:
            platos[_n(m.group(1).strip())] = int(m.group(2))
    return platos

def detectar_cocina(nombre, todos_platos):
    nombre_n = _n(str(nombre))
    # Excluidos explícitamente
    if nombre_n in EXCLUIR_CATEGORIA:
        return ''
    for cocina, kws in NOMBRE_KEYWORDS.items():
        if cocina == 'sin_categoria':
            continue
        # Categorías especiales no son cocinas — pero si el nombre matchea,
        # devolver '' para limpiar cualquier cocina_detectada anterior incorrecta
        if cocina in ('arroceria', 'marisqueria', 'asador'):
            if any(kw in nombre_n for kw in kws):
                return ''
            continue
        if any(kw in nombre_n for kw in kws):
            return cocina
    platos = _parsear_platos(todos_platos)
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
                    if mn >= MIN_MENCIONES:
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

def detectar_categoria(nombre, todos_platos, cocina_detectada):
    nombre_n = _n(str(nombre))
    # Excluidos explícitamente
    if nombre_n in EXCLUIR_CATEGORIA:
        return ''
    mapa_nombre = {
        'italiana': 'italiano', 'japonesa': 'japones', 'india': 'indio',
        'mexicana': 'mexicano', 'peruana': 'peruano', 'venezolana': 'venezolano',
        'argentina': 'argentino', 'tailandesa': 'tailandes', 'arabe': 'arabe',
        'vasca': 'taberna', 'madrileña': 'taberna', 'gallega': 'taberna',
        'asturiana': 'taberna', 'andaluza': 'taberna', 'fusion': 'fusion',
        'americana': 'hamburgueseria',
        'arroceria': 'arroceria', 'marisqueria': 'marisqueria', 'asador': 'asador',
    }
    for categoria, kws in NOMBRE_KEYWORDS.items():
        if categoria == 'sin_categoria':
            continue
        if any(kw in nombre_n for kw in kws):
            return mapa_nombre.get(categoria, categoria)
    platos = _parsear_platos(todos_platos)
    if platos:
        total = sum(platos.values())
        mejor, mejor_score = '', 0.0
        for categoria, defs in PLATOS_CATEGORIA.items():
            matches, menciones = [], 0
            min_p = MIN_PLATOS_MARISQUERIA if categoria == 'marisqueria' else MIN_PLATOS
            for d in defs:
                dn = _n(d)
                for pr, mn in platos.items():
                    if dn == pr or (len(dn) > 5 and dn in pr):
                        if mn >= MIN_MENCIONES:
                            matches.append(mn)
                            menciones += mn
                        break
            if len(matches) < min_p:
                continue
            if menciones / total < MIN_PROPORCION:
                continue
            score = sum(2 + math.log2(m) for m in matches)
            if score > mejor_score:
                mejor_score, mejor = score, categoria
        if mejor:
            return mejor
    mapa_cocina = {
        'italiana': 'italiano', 'japonesa': 'japones', 'india': 'indio',
        'mexicana': 'mexicano', 'peruana': 'peruano', 'venezolana': 'venezolano',
        'argentina': 'argentino', 'tailandesa': 'tailandes', 'arabe': 'arabe',
        'vasca': 'taberna', 'madrileña': 'taberna', 'gallega': 'taberna',
        'asturiana': 'taberna', 'andaluza': 'taberna', 'fusion': 'fusion',
        'americana': 'hamburgueseria',
    }
    return mapa_cocina.get(str(cocina_detectada), '')

df = pd.read_csv('analisis_restaurantes.csv')
df['cocina_detectada'] = df.apply(
    lambda r: detectar_cocina(str(r['nombre']), str(r['todos_platos'])), axis=1
)
df['categoria_carta'] = df.apply(
    lambda r: detectar_categoria(r['nombre'], r['todos_platos'], r['cocina_detectada']), axis=1
)

from collections import Counter
print('DISTRIBUCIÓN cocina_detectada:')
print(df['cocina_detectada'].replace('','(sin categoría)').value_counts().to_string())
print()
print('DISTRIBUCIÓN categoria_carta:')
print(df['categoria_carta'].replace('','(sin categoría)').value_counts().to_string())
print()
for cocina in sorted(df['cocina_detectada'].replace('','zzz').unique()):
    label = '' if cocina == 'zzz' else cocina
    filas = df[df['cocina_detectada'].replace('','zzz')==cocina]
    print(f'[{label or "SIN CATEGORÍA"}]')
    for _, r in filas.iterrows():
        print(f'  {r["nombre"]}: {str(r["todos_platos"])[:90]}')
    print()

df.to_csv('analisis_restaurantes.csv', index=False)
print('Guardado.')
