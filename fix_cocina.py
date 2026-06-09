import re, math, unicodedata, pandas as pd
from collections import defaultdict

def _n(s):
    s = s.lower().strip()
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')

NOMBRE_KEYWORDS = {
    'gallega':    ['galicia', 'gallego', 'gallega', 'montes de galicia', 'morgana'],
    'asturiana':  ['asturian', 'asturias', 'eiffel restaurante'],
    'vasca':      ['txirimiri', 'dantxari', 'txoko', 'euskal', 'juana la loca'],
    'andaluza':   ['gaditana', 'gaditano', 'sevill', 'andaluz'],
    'madrileña':  ['madril', 'casa toni', 'casa lucas', 'casa carola', 'angelita madrid',
                   'gastro', 'taberna la concha', 'la castela', 'meson molinero',
                   'taberna malaspina', 'el rincon del cabo', 'vinos de bellota',
                   'la descarada', 'sazon', 'cafe astral'],
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

NOMBRE_CATEGORIA = {
    'hamburgueseria': ['steakburger', 'steak burger', 'hamburgues', 'burger', 'burnout'],
    'arroceria':      ['arroceria', 'arrocería'],
    'japones':        ['sibuya', 'hotaru', 'miyama', 'ichikani', 'dokidoki',
                       'kaiten sushi', 'sr.ito', 'sakale'],
    'italiano':       ['trattoria', 'pizzeria', 'pizzart', 'mozzarell', 'napoli',
                       'piamonte', 'fusco', 'pastamore', 'malafemmena', 'malatesta',
                       'pulcinella', 'maruzzella', 'oliveto', 'piccola', 'davanti',
                       'bresca'],
    'indio':          ['indian', 'tandoori', 'bangalore', 'kathmandu', 'purnima',
                       'radhuni', 'curry masala', 'indian aroma'],
    'mexicano':       ['taco bar', 'mawey', 'el rey de los tacos', 'tacos &',
                       'la encomienda'],
    'peruano':        ['kausa', 'quispe', 'tampu', 'ronda 14'],
    'venezolano':     ['grama lounge'],
    'arabe':          ['hummuseria', 'beytna'],
    'argentino':      ['argentin', 'pampa beef', 'cabaña argentina', 'bayres',
                       'asado central', 'camoati', 'el social', 'mu! el placer',
                       'mu el placer'],
    'fusion':         ['diverxo', 'streetxo', 'dstage', 'bacira', 'coque',
                       'bestial', 'casa jaguar'],
    'tailandes':      ['bambúbox', 'bambubox'],
}

PLATOS_CATEGORIA = {
    'asador':      ['chuleton', 'chuletón', 'lechazo', 'entraña', 'lomo alto',
                    'costillar', 'secreto iberico', 'secreto ibérico',
                    'presa iberica', 'mollejas', 'cordero'],
    'marisqueria': ['percebes', 'vieiras', 'navajas', 'ostras', 'cigalas',
                    'bogavante', 'langosta', 'carabineros', 'centollo',
                    'buey de mar', 'zamburinas'],
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
}

# Umbral más alto para marisqueria — evitar falsos positivos con zamburiñas sueltas
MIN_PLATOS_MARISQUERIA = 3
MIN_PLATOS   = 2
MIN_MENCIONES = 2
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
    for cocina, kws in NOMBRE_KEYWORDS.items():
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

def detectar_categoria(nombre, todos_platos, cocina_detectada):
    nombre_n = _n(str(nombre))

    # Capa 1: nombre
    for categoria, kws in NOMBRE_CATEGORIA.items():
        if any(kw in nombre_n for kw in kws):
            return categoria

    # Capa 2: platos
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

    # Capa 3: heredar de cocina_detectada
    mapa = {
        'italiana': 'italiano', 'japonesa': 'japones', 'india': 'indio',
        'mexicana': 'mexicano', 'peruana': 'peruano', 'venezolana': 'venezolano',
        'argentina': 'argentino', 'tailandesa': 'tailandes', 'arabe': 'arabe',
        'vasca': 'taberna', 'madrileña': 'taberna', 'gallega': 'taberna',
        'asturiana': 'taberna', 'andaluza': 'taberna', 'fusion': 'fusion',
        'americana': 'hamburgueseria',
    }
    return mapa.get(str(cocina_detectada), '')

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

por_cat = defaultdict(list)
for _, r in df.iterrows():
    por_cat[r['categoria_carta']].append(r['nombre'])
for cat in sorted(por_cat):
    print(f'[{cat or "SIN CATEGORÍA"}]')
    for n in por_cat[cat]: print(f'  {n}')
    print()

df.to_csv('analisis_restaurantes.csv', index=False)
print('Guardado.')
