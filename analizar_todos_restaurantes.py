"""
Análisis masivo de reseñas — 125 restaurantes
Sentimiento: nlptown (local)
Platos: tokens + bigramas con filtro NO_PLATOS
Nombres: Gemini (solo para clasificar nombres de personas)
Normalización: Gemini (solo para corregir ortografía de platos)
Criterios: Gemini sobre fragmentos relevantes (mínimo de tokens)
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
import urllib.request as _ureq
import json as _json
import time as _time
warnings.filterwarnings('ignore')

# ── .env ──────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
CSV_PATH       = "resenas_unificadas.csv"
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "analisis_restaurantes.csv")
OUTPUT_RESENAS = os.path.join(OUTPUT_DIR, "analisis_restaurantes_resenas.csv")
MODELO_HF      = "nlptown/bert-base-multilingual-uncased-sentiment"
MIN_MENCIONES  = 10
MIN_CONFIANZA  = 0.50
BATCH_SIZE     = 32

# ── Modo económico ────────────────────────────────────────────────────────────
# True  → desactiva normalización de platos y clasificación de personal.
#          Ahorra ~2 llamadas Gemini por restaurante. Usar cuando los platos
#          ya están bien del procesado anterior y solo quieres añadir columnas
#          nuevas (criterios, dimensiones, todos_platos).
# False → procesado completo con normalización y nombres (más caro pero mejor).
MODO_ECONOMICO = True

# Palabras de negación — usadas para filtrar fragmentos antes de Gemini
NEGACIONES = {'no','sin','nunca','tampoco','ni','jamás','jamas','nada'}

STOP_ES = set(stopwords.words('spanish')) | {
    'si','así','tan','ser','año','vez','bien','muy','más','menos',
    'todo','toda','todos','cada','este','esta','estos','estas',
    'aún','ahí','allí','aquí','hay','era','fue','han','has',
    'hemos','nos','les','lo','la','las','los','del','al',
    'un','una','unos','unas','su','sus','mi','mis','se','me',
    'te','le','que','de','en','y','a','el','es','por','con',
    'para','como','no','pero','o','si','cuando','donde','aunque',
}

# Palabras que NO son platos — amplio y definitivo
NO_PLATOS = {
    'restaurante','comida','cocina','servicio','trato','atención','experiencia','lugar',
    'sitio','personal','camarero','camarera','ambiente','precio','calidad','carta',
    'mesa','plato','platos','producto','productos','cocina','local','establecimiento',
    'buena','bueno','buen','buenos','buenas','rico','rica','ricos','ricas',
    'riquísimo','delicioso','deliciosa','deliciosos','deliciosas',
    'exquisito','exquisita','espectacular','increíble','perfecto','perfecta',
    'impecable','excepcional','excelente','fantástico','fantástica',
    'maravilloso','maravillosa','inmejorable','estupendo','estupenda',
    'sobresaliente','alucinante','brutal','genial','fenomenal',
    'recomendable','insuperable','único','única','especial','auténtico',
    'fresco','fresca','sabroso','sabrosa','increible','magnifico',
    'deliciosa','mejor','peor','igual','mismo','misma',
    'rapido','rapida','amable','amables','atento','atenta','atentos','atentas',
    'agradable','agradables','simpatico','simpatica','profesional','profesionales',
    'eficiente','eficientes','educado','educada','cercano','cercana',
    'súper','super','muy','mucho','mucha','muchos','muchas','poco','poca',
    'siempre','nunca','también','además','realmente','totalmente',
    'absolutamente','simplemente','especialmente','principalmente',
    'gracias','encantado','encantada','satisfecho','satisfecha',
    'volver','volveremos','volvería','repetiremos','repetir',
    'recomiendo','recomendamos','recomendable','recomendado','recomendada',
    'pedimos','pedí','tomamos','probamos','atendió','atendieron',
    'encantó','gustó','quedamos','salimos','fuimos','vinimos',
    'visita','primera','segunda','tercera','última','próxima',
    'ocasión','celebración','cumpleaños','aniversario','cena','almuerzo',
    'reserva','mesa','cuenta','nota','propina',
    'acogedor','acogedora','principal','destacar','destacado','destacada',
    'verdad','variedad','variedad de','tiempo','chicos','chicas',
    'comer','tomar','pedir','puse','puse','salir','venir','hacer',
    'gente','personas','novio','novia','marido','mujer','pareja',
    'brunch','brunch de','momento','momentos','sabor','sabores',
    'cantidad','cantidades','racion','raciones','ración','raciones',
    'presentacion','presentación','vista','vistas',
    'madrid','barcelona','españa','galicia','india','italia','japón',
    'mexico','peru','venezuela','colombia','cuba','argentina',
    'familia','amigos','pareja','grupo','equipo','compañeros',
    'menú','degustación','carta','precios','euros','coste',
    'duda','acierto','momento','punto','nivel','tipo','clase',
    'detalle','detalles','nada','algo',
    'sin duda','por supuesto','desde luego','muy recomendable',
    'plato principal','gran plato','buen plato',
    'calidad precio','buena relacion',
    'comida india','comida italiana','comida japonesa','comida china',
    'comida mexicana','comida peruana','comida venezolana','comida colombiana',
    'comida española','comida francesa','comida griega','comida árabe',
    'comida arabe','comida turca','comida americana','comida internacional',
    'comida casera','comida tradicional','comida fusion','comida fusión',
    'comida rapida','comida rápida','comida callejera',
    'comida rica','comida muy rica','comida buena','comida muy buena',
    'comida excelente','comida deliciosa','comida riquísima','comida increíble',
    'comida espectacular','comida fantástica','comida perfecta','comida inmejorable',
    'comida abundante','comida escasa','comida fresca','comida caliente','comida fría',
    'cocina india','cocina italiana','cocina japonesa','cocina china',
    'cocina mexicana','cocina peruana','cocina venezolana','cocina española',
    'cocina francesa','cocina mediterránea','cocina mediterranea',
    'cocina tradicional','cocina fusion','cocina fusión','cocina casera',
    'cocina rica','cocina muy rica','cocina buena','cocina excelente',
}

PLATOS_WHITELIST = {
    'pulpo','chuletón','chuleton','cochinillo','lechazo','cordero','rabo','carrillera',
    'bacalao','merluza','rodaballo','lubina','dorada','rape','salmón','salmon',
    'atún','atun','bonito','anchoas','gambas','langostinos','calamares','chipirones',
    'mejillones','almejas','berberechos','navajas','percebes','langosta','bogavante',
    'sepia','cocochas',
    'cachopo','cachopos','fabada','callos','cocochas','kokotxas','marmitako',
    'pimientos rellenos','pisto manchego','migas','gazpacho manchego',
    'papas arrugadas','mojo','escudella','pa amb tomàquet',
    'porrusalda','txangurro','angulas','kokotxas al pil pil',
    'secreto ibérico','secreto','presa ibérica','presa','pluma ibérica','pluma',
    'carrillada','oreja','manitas','callos a la madrileña',
    'albóndigas','albondigas','rabo de toro','rabo toro',
    'jamón','jamon','lomo','chorizo','morcilla','salchichón','salchicon','sobrasada',
    'cecina','foie','mollejas',
    'paella','arroz','risotto','fideuá','fideua','pasta','espaguetis','lasaña','lasana',
    'carbonara','penne','ravioli','tagliatelle',
    'croquetas','croqueta','patatas bravas','tortilla','gazpacho','salmorejo',
    'ensaladilla','berenjenas','pimientos','pisto','ratatouille','hummus',
    'guacamole','nachos','bruschetta','focaccia','tostas',
    'tikka masala','biryani','naan','samosa','curry','korma','dal','tandoori',
    'arepas','arepa','pabellón','pabellon','cachapas','tequeños','tequenos',
    'empanadas','empanada','ceviche','tiradito','lomo saltado','causa','anticuchos',
    'sushi','sashimi','ramen','gyozas','edamame','tempura','miso',
    'tacos','burrito','quesadilla','fajitas',
    'tarta','tartaleta','coulant','brownie','tiramisú','tiramisu','panna cotta',
    'flan','crema catalana','helado','mousse','cheesecake','mochi',
    'churros','soufflé','souffle','crepe','waffle','suspiro',
    'sangría','sangria','mojito','margarita','caipirinha','negroni','martini',
    'buñuelos','bunuelos','buñuelos de bacalao',
    'paella de mariscos','paella de marisco',
    # Platos asturianos y otras incorporaciones detectadas en producción
    'cachopo','cachopu','cachopo de ternera','cachopo relleno','cachopo casero',
    'fabada asturiana','fabada','pote asturiano','pote gallego','oricios',
    'tortos','torto','frixuelos','verdinas','verdinas con marisco',
    'compango','cabrales','queso cabrales','afuega','gamonedo',
    'sidra asturiana','botillo','chosco','cecina asturiana',
    'merluza asturiana','pixín','pixin','centollo asturiano',
    # Platos vascos adicionales
    'pintxos','pintxo','gilda','pil pil',
    # Platos madrileños
    'cocido madrileno','cocido madrileño','cocido','cocido completo',
    'callos madrilenos','callos a la madrilena','huevos rotos','huevos estrellados',
    'bocadillo de calamares','bocata de calamar','soldaditos de pavía',
    'patatas a la importancia','migas',
    # Postres y dulces adicionales
    'arroz con leche','leche frita','bizcocho','magdalenas',
    'torrijas','rosquillas','buñuelos de viento',
    # Cortes de carne
    'chuletón de buey','entrecot','solomillo','filete','picaña',
    'entraña','tira de asado','costillar',
    # Mariscos y pescados adicionales
    'zamburiñas','nécoras','centollo','buey de mar',
    'boquerones','boquerones en vinagre','boquerones fritos','boquerones aceite',
    'gambas al ajillo','gambas plancha','gambas pil pil',
    'lomos de bacalao','anchoas en salazón','anchoas',
    'carabineros','cigalas','ostras','navajas plancha',
    # Platos con adjetivo geográfico que el extractor perdia
    'cocido madrileno','fabada asturiana','lacon con grelos','caldo gallego',
    'bacalao vizcaina','bacalao al pil pil','merluza vasca',
    'pinchos','pintxo','pinchos morunos',
    # Bebidas y cócteles adicionales
    'vermut','vermu','vermú','rebujito','tinto de verano',
    'agua de valencia','txakoli','sidra',
}

# ── Falsos positivos confirmados — se rechazan SIEMPRE, ignorando la caché ──
# Añade aquí cualquier término que el script haya aprobado incorrectamente.
# Tienen prioridad absoluta sobre PLATOS_WHITELIST y sobre _cache_es_plato.
PLATOS_BLACKLIST = {
    # Frases de valoración genérica detectadas como falsos positivos
    'siempre la comida',
    'la comida siempre',
    'toda la comida',
    'muy buena comida',
    'buenísima comida',
    'la mejor comida',
    'toda la carta',
    'todo muy bueno',
    'todo estuvo',
    'todo estaba',
    'todo rico',
    'todo bueno',
    'todo excelente',
    'siempre bueno',
    'siempre bien',
    'muy recomendable',
    'muy bien todo',
    # Experiencias / descripción del servicio
    'primera vez',
    'segunda vez',
    'primera visita',
    'próxima visita',
    'muy buen servicio',
    'buen servicio',
    'gran servicio',
    'muy buen trato',
    'muy buena atención',
    # Detectados en producción — frases de experiencia, no platos
    'verdad que sushi','sushi más divino','sushi de calidad','sushi de madrid',
    'comer sushi','gusta el sushi','barra de sushi','restaurante de sushi',
    'solo sushi','kaiten sushi','marca el sushi','sushi rico','piezas de sushi',
    'local muy bonito','local bonito','bonito y limpio','sitio bonito',
    'sitio muy bonito','restaurante es muy bonito','restaurante muy bonito',
    'local es muy bonito','pasta y la pizza','pasta como la pizza',
    'pasta y dos','pasta y pizza','platos de pasta','plato de pasta',
    'pedimos nachos','pedimos arroz','pedimos croquetas','pedimos los nachos',
    'pedimos unas gyozas','pedimos una pasta','pedí una paella',
    'probamos arroz','probamos unos tagliatelle',
    'agradable la paella','paella deliciosa','recomendamos las croquetas',
    'especialmente el risotto','especialmente la pasta','especialmente las zamburiñas',
    'pedir el risotto','cenado un lomo','acá la pasta',
    'razonables el bacalao','puerros el bacalao',
    'maria un la pasta','guacamole lo hacen',
    'postre la tarta','postre tarta','postre torrijas','postre helado',
    'cordero excepcional','cordero espectacular',
    'sashimi espectacular','arroz muy bueno','pasta muy buena',
    'empanadas y tostas','pasta fuera de carta','pasta en la rueda',
    'igual que los tacos','igual que los t',
    'segunda entraña','segundo entraña',
    'tandoori era muy bueno','carbonara estuvo deliciosa',
    'fresca y el tiramisu','tiramisu estaba excelente',
    'bourbon y el costillar','costillar deliciosossss',
    'boloñesa el tiramisú','tiramisú y el helado',
    'arroz señorito','señorito y zamburiñas','zamburiñas nos encantaron',
    'probamos arroz',
    # Añade aquí los que vayas encontrando en tu dataset
}

PATRONES_NO_PLATO = re.compile(
    r'^(muy|gran|super|súper|bien|mal|sin|con|para|como|desde|hasta|'
    r'nuestro|nuestra|vuestro|vuestra|todo|toda|todos|todas|'
    r'primer|primera|segundo|segunda|otro|otra|cada|mismo|misma)$'
    r'|'
    r'(mente|ísimo|ísima|ísimos|ísimas|ción|sión)$'
)

# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE COCINA POR PLATOS
# ─────────────────────────────────────────────────────────────────────────────

PLATOS_POR_COCINA = {
    'gallega':    ['empanada gallega', 'percebes', 'vieiras', 'caldo gallego', 'lacon', 'filloas', 'pote gallego', 'zorza'],
    'asturiana':  ['cachopo', 'cachopu', 'fabada asturiana', 'fabada', 'oricios', 'cabrales', 'tortos', 'frixuelos', 'verdinas'],
    'vasca':      ['pintxos', 'pintxo', 'gilda', 'kokotxas', 'cocochas', 'txangurro', 'marmitako', 'txakoli', 'pil pil', 'angulas'],
    'andaluza':   ['pescaito frito', 'tortillitas de camarones', 'ajoblanco', 'berenjenas con miel', 'espinacas con garbanzos', 'cola de toro'],
    'madrileña':  ['cocido madrileño', 'cocido madrileno', 'callos a la madrileña', 'bocadillo de calamares', 'soldaditos de pavia'],
    'italiana':   ['carbonara', 'cacio e pepe', 'amatriciana', 'ossobuco', 'panna cotta', 'tiramisú', 'tiramisu',
                   'tagliatelle', 'pappardelle', 'gnocchi', 'cannoli', 'arancini', 'burrata', 'lasaña', 'lasana', 'pizza', 'bruschetta', 'focaccia'],
    'peruana':    ['lomo saltado', 'lomo salteado', 'causa', 'causa limeña', 'anticuchos', 'aji de gallina', 'leche de tigre', 'chaufa', 'tiradito'],
    'japonesa':   ['ramen', 'sushi', 'sashimi', 'nigiri', 'gyozas', 'gyoza', 'tempura', 'udon', 'mochi', 'edamame', 'yakitori', 'takoyaki'],
    'india':      ['tikka masala', 'biryani', 'naan', 'samosa', 'korma', 'dal', 'tandoori', 'butter chicken', 'palak paneer'],
    'mexicana':   ['tacos', 'burrito', 'quesadilla', 'fajitas', 'enchilada', 'pozole', 'carnitas', 'chilaquiles', 'tamales'],
    'venezolana': ['arepa', 'arepas', 'pabellon', 'cachapa', 'hallaca', 'tequeños', 'mandocas', 'caraotas'],
    'argentina':  ['entraña', 'mollejas', 'chorizo criollo', 'empanadas argentinas', 'empanadas criollas', 'lomo alto', 'chimichurri', 'provoleta'],
    'arabe':      ['shawarma', 'falafel', 'tabule', 'baba ganoush', 'labneh', 'shakshuka', 'kibbeh', 'couscous'],
    'americana':  ['smash burger', 'pulled pork', 'costillas bbq', 'mac and cheese', 'chicken wings', 'brisket'],
    'griega':     ['gyros', 'souvlaki', 'moussaka', 'spanakopita', 'baklava', 'dolmades'],
    'china':      ['dim sum', 'wonton', 'pato pekin', 'chow mein', 'dumplings'],
    'tailandesa': ['pad thai', 'tom yum', 'massaman', 'satay'],
    'francesa':   ['foie gras', 'confit de pato', 'magret', 'bouillabaisse', 'escargots', 'cassoulet'],
    'colombiana': ['bandeja paisa', 'ajiaco', 'sancocho'],
}

_NOMBRE_KEYWORDS = {
    'gallega':    ['galicia', 'gallego', 'gallega'],
    'asturiana':  ['asturian', 'asturias'],
    'vasca':      ['txirimiri', 'dantxari', 'txoko', 'euskal'],
    'andaluza':   ['gaditana', 'gaditano', 'sevill', 'andaluz'],
    'madrileña':  ['madril'],
    'argentina':  ['argentin', 'pampa beef', 'cabaña argentina', 'bayres', 'asado central', 'camoati'],
    'italiana':   ['trattoria', 'pizzeria', 'pizzart', 'mozzarell', 'napoli', 'piamonte',
                   'fusco', 'pastamore', 'malafemmena', 'malatesta', 'pulcinella',
                   'maruzzella', 'oliveto', 'piccola', 'davanti', 'bresca'],
    'japonesa':   ['sibuya', 'hotaru', 'miyama', 'ichikani', 'dokidoki', 'kaiten sushi', 'sr.ito', 'sakale'],
    'india':      ['indian', 'tandoori', 'bangalore', 'kathmandu', 'purnima',
                   'radhuni', 'curry masala', 'indian aroma'],
    'mexicana':   ['taco bar', 'mawey', 'el rey de los tacos', 'tacos &'],
    'peruana':    ['kausa', 'quispe', 'tampu', 'ronda 14'],
    'venezolana': ['grama lounge'],
    'arabe':      ['hummuseria', 'beytna'],
    'americana':  ['steakburger', 'steak burger', 'hamburgues', 'burnout', 'brew wild'],
    'fusion':     ['diverxo', 'streetxo', 'dstage', 'bacira', 'coque', 'bestial', 'casa jaguar'],
}

def _detectar_cocina_restaurante(todos_platos_str: str, nombre_restaurante: str = '') -> str:
    import unicodedata as _ud, re as _re, math as _math
    def _n(s):
        s = s.lower().strip()
        s = _ud.normalize('NFD', s)
        return ''.join(c for c in s if _ud.category(c) != 'Mn')

    # Capa 1: nombre
    nombre_n = _n(nombre_restaurante)
    for cocina, kws in _NOMBRE_KEYWORDS.items():
        if any(kw in nombre_n for kw in kws):
            return cocina

    # Capa 2: platos exclusivos — si no hay señal clara, devuelve ''
    if not todos_platos_str or str(todos_platos_str).strip() in ('', 'nan'):
        return ''
    platos = {}
    for p in str(todos_platos_str).split(','):
        m = _re.match(r'^(.+?)\((\d+)\)$', p.strip())
        if m:
            platos[_n(m.group(1).strip())] = int(m.group(2))
    if not platos:
        return ''
    total = sum(platos.values())
    mejor, mejor_score = '', 0.0
    for cocina, defs in PLATOS_POR_COCINA.items():
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
        score = sum(2 + _math.log2(m) for m in matches)
        if score > mejor_score:
            mejor_score, mejor = score, cocina
    return mejor


# ─────────────────────────────────────────────────────────────────────────────
# Cache persistente
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_PLATOS_PATH   = os.path.join(OUTPUT_DIR, ".cache_platos_gemini.json")
_CACHE_NORM_PATH     = os.path.join(OUTPUT_DIR, ".cache_norm_gemini.json")
_CACHE_CRITERIOS_PATH= os.path.join(OUTPUT_DIR, ".cache_criterios_gemini.json")
_cache_nombres_path  = os.path.join(OUTPUT_DIR, ".cache_nombres_gemini.json")

_cache_es_plato:      dict = {}
_cache_normalizacion: dict = {}
_cache_criterios:     dict = {}  # clave: id_restaurante → dict de criterios
_cache_nombres:       dict = {}

def _cargar_cache_platos():
    global _cache_es_plato, _cache_normalizacion, _cache_nombres, _cache_criterios
    if os.path.exists(_CACHE_PLATOS_PATH):
        try:
            with open(_CACHE_PLATOS_PATH, 'r', encoding='utf-8') as f:
                loaded = _json.load(f)
            _cache_es_plato = {k: v for k, v in loaded.items() if v is True}
            # Purgar de la caché cualquier término que esté en la blacklist
            n_antes_purga = len(_cache_es_plato)
            _NO_EN_PLATO_CACHE = {
                'pedi','pedimos','probe','probamos','tome','tomamos','cenamos',
                'comimos','recomiendo','recomendamos','trajeron',
                'divino','divina','genial','increible','increíble','espectacular',
                'buenísimo','buenísima','riquísimo','riquísima','agradable',
                'bonito','bonita','solo','sólo','verdad','gusta',
                'restaurante','local','sitio','barra','rueda','marca',
            }
            def _contiene_no_plato(k):
                return any(p in _NO_EN_PLATO_CACHE for p in k.lower().split())
            _cache_es_plato = {k: v for k, v in _cache_es_plato.items()
                               if k not in PLATOS_BLACKLIST
                               and not _contiene_no_plato(k)}
            n_purgados = n_antes_purga - len(_cache_es_plato)
            n_total = len(loaded); n_kept = len(_cache_es_plato)
            msg = f"Cache clasificación: {n_kept} aprobados ({n_total - n_kept} rechazados descartados para re-evaluar)"
            if n_purgados:
                msg += f" · {n_purgados} purgados por blacklist"
            print(msg + ".")
        except Exception:
            _cache_es_plato = {}
    if os.path.exists(_CACHE_NORM_PATH):
        try:
            with open(_CACHE_NORM_PATH, 'r', encoding='utf-8') as f:
                _cache_normalizacion = _json.load(f)
            print(f"Cache normalización: {len(_cache_normalizacion)} conjuntos conocidos.")
        except Exception:
            _cache_normalizacion = {}
    if os.path.exists(_CACHE_CRITERIOS_PATH):
        try:
            with open(_CACHE_CRITERIOS_PATH, 'r', encoding='utf-8') as f:
                _cache_criterios = _json.load(f)
            print(f"Cache criterios:     {len(_cache_criterios)} restaurantes conocidos.")
        except Exception:
            _cache_criterios = {}
    if os.path.exists(_cache_nombres_path):
        try:
            with open(_cache_nombres_path, 'r', encoding='utf-8') as f:
                _cache_nombres.update(_json.load(f))
            print(f"Cache nombres:       {len(_cache_nombres)} términos conocidos.")
        except Exception:
            pass

def _guardar_cache_platos():
    try:
        with open(_CACHE_PLATOS_PATH, 'w', encoding='utf-8') as f:
            _json.dump(_cache_es_plato, f, ensure_ascii=False, indent=2)
        with open(_CACHE_NORM_PATH, 'w', encoding='utf-8') as f:
            _json.dump(_cache_normalizacion, f, ensure_ascii=False, indent=2)
        with open(_CACHE_CRITERIOS_PATH, 'w', encoding='utf-8') as f:
            _json.dump(_cache_criterios, f, ensure_ascii=False, indent=2)
        with open(_cache_nombres_path, 'w', encoding='utf-8') as f:
            _json.dump(_cache_nombres, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  No se pudo guardar cache: {e}")

# ─────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────
_GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_MODEL = "gemini-2.5-flash"

# ── Contador de tokens y coste estimado ──────────────────────────────────────
# Precios gemini-2.5-flash (junio 2025): input $0.075/1M tokens, output $0.30/1M tokens
_tok_input  = 0
_tok_output = 0

def _registrar_uso(prompt_len_chars, output_len_chars):
    """Estima tokens (1 token ≈ 4 chars) y acumula coste."""
    global _tok_input, _tok_output
    _tok_input  += prompt_len_chars  // 4
    _tok_output += output_len_chars  // 4

def coste_estimado():
    """Devuelve coste estimado en EUR (aprox, tasa 1 USD = 0.92 EUR)."""
    usd = (_tok_input / 1_000_000 * 0.075) + (_tok_output / 1_000_000 * 0.30)
    return usd * 0.92

def _gemini_call(url, data, retries=4):
    for intento in range(retries):
        texto_resp = ""
        try:
            req  = _ureq.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
            resp = _ureq.urlopen(req, timeout=20)
            result = _json.loads(resp.read())
            parts  = result["candidates"][0]["content"]["parts"]
            for part in parts:
                if part.get("text","").strip():
                    texto_resp = part["text"].strip()
            _registrar_uso(len(data), len(texto_resp))
            return texto_resp
        except Exception as e:
            if any(c in str(e) for c in ['503','429','500']) and intento < retries-1:
                espera = 2 ** (intento + 1)
                print(f"      [Gemini] reintentando en {espera}s...")
                _time.sleep(espera)
            else:
                return ""
    return ""

def _gemini_json(prompt, max_tokens=200):
    if not _GEMINI_KEY:
        return {}
    data = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1/models/{_GEMINI_MODEL}:generateContent?key={_GEMINI_KEY}"
    texto = _gemini_call(url, data)
    _time.sleep(0.5)
    if not texto:
        return {}
    texto = texto.replace("\u201c",'"').replace("\u201d",'"')
    texto = texto.replace("\u2018","'").replace("\u2019","'")
    texto = texto.replace("```json","").replace("```","").strip()
    inicio = texto.find("{"); fin = texto.rfind("}")+1
    if inicio >= 0 and fin > inicio:
        texto = texto[inicio:fin]
    try:
        return _json.loads(texto)
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# CRITERIOS — keywords de señal y extracción de fragmentos
# ─────────────────────────────────────────────────────────────────────────────

# Keywords de señal: palabras que indican que la reseña PUEDE mencionar el criterio.
# Más estrictas que DIMENSIONES_KEYWORDS — solo palabras con alta especificidad.
CRITERIOS_SIGNAL = {
    # ── Criterios originales ──────────────────────────────────────────────────
    'ninos':               ['niño','niños','niña','niñas','bebé','bebe','infantil',
                            'sillita','trona','peques','pequeños','crío','crios','chaval'],
    'mascotas':            ['perro','perros','mascota','mascotas','peludo','peludos',
                            'admiten perros','dog','pet'],
    'terraza':             ['terraza','terrazas','exterior','al aire libre',
                            'patio','veladores','parasol'],
    'vistas':              ['vistas','panorámica','panoramica','azotea','rooftop',
                            'mirador','skyline','horizonte'],
    'musica_directo':      ['música en directo','música directo','concierto',
                            'actuación','actuacion','banda','grupo en vivo','en vivo',
                            'jazz','flamenco en directo'],
    'romantico':           ['romántico','romantico','romántica','romantica',
                            'íntimo','intimo','íntima','intima',
                            'cena romántica','velas'],
    # ── Criterios nuevos ─────────────────────────────────────────────────────
    'buen_postre':         ['postre','postres','tarta','helado','tiramisú','tiramisu',
                            'mousse','coulant','brownie','cheesecake','flan',
                            'crema catalana','panna cotta'],
    'precio_calidad':      ['calidad precio','calidad-precio','relacion calidad',
                            'relación calidad','precio razonable','precio asequible',
                            'muy economico','muy económico','precio justo',
                            'buena relacion','buena relación'],
    'grupos_grandes':      ['grupo grande','grupos grandes','celebracion','celebración',
                            'cumpleanos','cumpleaños','evento privado','reserva para grupo',
                            'cena de empresa','comida de empresa','gran grupo'],
    'vegano_vegetariano':  ['vegano','vegana','vegetariano','vegetariana',
                            'opciones veganas','opciones vegetarianas',
                            'plant based','sin carne','menú vegano'],
    'sin_gluten':          ['sin gluten','celiaco','celiaca','celíaco','celíaca',
                            'gluten free','opcion sin gluten','opción sin gluten'],
}

# Descripción corta para el prompt de Gemini
CRITERIOS_DESC = {
    'ninos':               'apto para niños o familias con bebés/niños pequeños',
    'mascotas':            'admite mascotas o perros',
    'terraza':             'tiene terraza o espacio exterior disponible',
    'vistas':              'tiene vistas panorámicas, azotea o mirador',
    'musica_directo':      'ofrece música en directo, conciertos o actuaciones',
    'romantico':           'ambiente romántico o íntimo',
    'buen_postre':         'destacan los postres según los clientes',
    'precio_calidad':      'buena relación calidad-precio mencionada explícitamente',
    'grupos_grandes':      'apto para grupos grandes, celebraciones o eventos',
    'vegano_vegetariano':  'tiene opciones veganas o vegetarianas claras',
    'sin_gluten':          'tiene opciones sin gluten o apto para celíacos',
}

def _extraer_fragmentos(serie_reviews, keywords, ventana=40):
    """
    Para cada reseña que contenga algún keyword, extrae la frase de contexto
    (ventana caracteres antes y después del keyword).
    Devuelve lista de fragmentos únicos, máximo 8.
    """
    fragmentos = []
    vistos = set()
    for resena in serie_reviews:
        texto = str(resena)
        texto_lower = texto.lower()
        for kw in keywords:
            idx = texto_lower.find(kw)
            if idx == -1:
                continue
            inicio = max(0, idx - ventana)
            fin    = min(len(texto), idx + len(kw) + ventana)
            # Recortar en límites de palabra
            if inicio > 0:
                inicio = texto.rfind(' ', 0, inicio) + 1
            if fin < len(texto):
                fin = texto.find(' ', fin)
                if fin == -1:
                    fin = len(texto)
            frag = texto[inicio:fin].strip()
            if frag and frag not in vistos:
                vistos.add(frag)
                fragmentos.append(frag)
            if len(fragmentos) >= 8:
                return fragmentos
    return fragmentos

def _fragmento_tiene_negacion(fragmento, keyword):
    """
    Comprueba si el keyword en el fragmento está precedido de negación
    (ventana de 4 palabras antes).
    """
    frag_lower = fragmento.lower()
    idx = frag_lower.find(keyword)
    if idx == -1:
        return False
    contexto_antes = frag_lower[max(0, idx-30):idx]
    palabras_antes = contexto_antes.strip().split()[-4:]
    return any(neg in palabras_antes for neg in NEGACIONES)

def clasificar_criterios_gemini(rid: int, nombre: str, serie_reviews) -> dict:
    """
    Clasifica los 6 criterios cualitativos usando Gemini solo sobre fragmentos
    relevantes extraídos de las reseñas. Mínimo de tokens.

    Estrategia:
    1. Para cada criterio, extraer fragmentos que contienen keywords de señal.
    2. Filtrar fragmentos con negación obvia (heurística rápida sin coste).
    3. Si no hay fragmentos → False sin llamar a Gemini.
    4. Si hay fragmentos → una sola llamada a Gemini con TODOS los criterios
       que tienen fragmentos (no una llamada por criterio).
    5. Cachear resultado por id_restaurante.
    """
    clave_cache = str(rid)
    if clave_cache in _cache_criterios:
        cached = _cache_criterios[clave_cache]
        # Recalculate frases even on cache hit (no Gemini cost)
        frases_cache = {}
        for criterio, keywords in CRITERIOS_SIGNAL.items():
            if cached.get(criterio):
                frags = _extraer_fragmentos(serie_reviews, keywords)
                frags_ok = [f for f in frags
                            if not any(_fragmento_tiene_negacion(f, kw) for kw in keywords)]
                if frags_ok:
                    if _GEMINI_KEY:
                        desc = CRITERIOS_DESC.get(criterio, criterio)
                        frases_cache[criterio] = _parafrasear_frases(frags_ok[:2], desc)
                    else:
                        frases_cache[criterio] = ' | '.join(frags_ok[:2])
        return cached, frases_cache

    # Resultado por defecto
    resultado = {c: False for c in CRITERIOS_SIGNAL}

    # ── Paso 1: extraer fragmentos por criterio ───────────────────────────────
    fragmentos_por_criterio = {}
    for criterio, keywords in CRITERIOS_SIGNAL.items():
        frags = _extraer_fragmentos(serie_reviews, keywords)
        # Filtro heurístico de negaciones: descartar fragmentos claramente negativos
        frags_ok = [f for f in frags
                    if not any(_fragmento_tiene_negacion(f, kw) for kw in keywords)]
        if frags_ok:
            fragmentos_por_criterio[criterio] = frags_ok

    # ── Paso 2: si no hay ningún fragmento, devolver todo False sin llamar a Gemini
    if not fragmentos_por_criterio:
        _cache_criterios[clave_cache] = resultado
        return resultado, {}

    # ── Paso 3: sin Gemini, usar heurística — si hay fragmentos sin negación → True
    if not _GEMINI_KEY:
        for criterio in fragmentos_por_criterio:
            resultado[criterio] = True
        frases = {c: ' | '.join(fragmentos_por_criterio[c][:2]) for c in fragmentos_por_criterio}
        _cache_criterios[clave_cache] = resultado
        return resultado, frases

    # ── Paso 4: construir prompt compacto con todos los criterios en UNA llamada
    secciones = []
    for criterio, frags in fragmentos_por_criterio.items():
        desc = CRITERIOS_DESC[criterio]
        frags_texto = ' | '.join(f'"{f}"' for f in frags[:5])  # máx 5 fragmentos
        secciones.append(f'- {criterio} ({desc}):\n  {frags_texto}')

    criterios_lista = list(fragmentos_por_criterio.keys())
    vals_default = ", ".join(f'"{c}": false' for c in criterios_lista)
    prompt = (
        f'Rest: {nombre}\n'
        + '\n'.join(secciones)
        + '\ntrue=confirmado positivamente en los fragmentos. false=duda/negación.\n'
        f'JSON SOLO: {{{vals_default}}}'
    )

    parsed = _gemini_json(prompt, max_tokens=100)


    for criterio in criterios_lista:
        val = parsed.get(criterio)
        if isinstance(val, bool):
            resultado[criterio] = val
        elif isinstance(val, str):
            resultado[criterio] = val.lower() == 'true'

    # Guardar también los fragmentos que justifican cada criterio True
    # (máx 2 por criterio, separados por " | ") — coste cero, ya están calculados
    frases = {}
    for criterio, frags in fragmentos_por_criterio.items():
        if resultado.get(criterio):
            if _GEMINI_KEY:
                desc = CRITERIOS_DESC.get(criterio, criterio)
                frases[criterio] = _parafrasear_frases(frags[:2], desc)
            else:
                frases[criterio] = ' | '.join(frags[:2])

    _cache_criterios[clave_cache] = resultado
    return resultado, frases


def _gemini_texto(prompt: str, max_tokens: int = 150) -> str:
    """Llamada a Gemini para texto libre (no JSON). Ahorra tokens con prompts compactos."""
    if not _GEMINI_KEY:
        return ""
    data = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1/models/{_GEMINI_MODEL}:generateContent?key={_GEMINI_KEY}"
    texto = _gemini_call(url, data)
    _time.sleep(0.3)
    return texto.strip()


def _parafrasear_frases(fragmentos: list[str], contexto: str = "") -> str:
    """
    Recibe fragmentos literales de reseñas y devuelve 1-2 frases parafraseadas
    en lenguaje natural limpio, sin identificar a nadie ni reproducir texto literal.
    Coste: ~80 tokens input + 60 output por llamada.
    """
    if not fragmentos or not _GEMINI_KEY:
        return ""
    # Truncar fragmentos para minimizar tokens
    frags_txt = " | ".join(f[:120] for f in fragmentos[:3])
    prompt = (
        f"Reescribe en 1-2 frases naturales y limpias lo que expresan estas opiniones "
        f"de clientes sobre {contexto}. Sin comillas, sin nombres propios, "
        f"sin reproducir el texto original. Solo la idea principal.\n"
        f"Opiniones: {frags_txt}\nFrases:"
    )
    return _gemini_texto(prompt, max_tokens=80)


def _generar_resenas_destacadas(df_reviews, nombre_restaurante: str) -> str:
    """
    Selecciona las 3 reseñas más positivas (5 estrellas, más largas) y
    las parafrasea con Gemini en frases limpias separadas por ' | '.
    Privacidad: no se guarda ni reproduce ningún texto literal.
    """
    if not _GEMINI_KEY or df_reviews is None or len(df_reviews) == 0:
        return ""

    # Seleccionar reseñas positivas largas
    col_stars = 'estrellas' if 'estrellas' in df_reviews.columns else 'Valoracion'
    df_pos = df_reviews[df_reviews[col_stars] >= 4].copy()
    df_pos['len'] = df_pos['Review'].astype(str).apply(len)
    df_pos = df_pos.sort_values(['len'], ascending=False).head(5)

    if len(df_pos) == 0:
        return ""

    fragmentos = df_pos['Review'].astype(str).tolist()[:3]
    frags_txt = " | ".join(f[:150] for f in fragmentos)

    prompt = (
        f"Resume en 3 frases cortas y naturales las mejores opiniones de clientes "
        f"sobre el restaurante {nombre_restaurante}. Cada frase separada por ' | '. "
        f"Sin comillas, sin nombres propios, sin reproducir texto literal. "
        f"Solo los aspectos más destacados.\n"
        f"Opiniones: {frags_txt}\nResumen:"
    )
    return _gemini_texto(prompt, max_tokens=120)


def filtrar_platos_con_gemini(candidatos: list, nombre_restaurante: str = "") -> list:
    if not candidatos:
        return candidatos
    resultado = []
    por_consultar = []
    for nombre, cuenta in candidatos:
        # Blacklist tiene prioridad absoluta — rechazar siempre, actualizar caché
        if nombre in PLATOS_BLACKLIST:
            _cache_es_plato[nombre] = False
            continue
        if nombre in PLATOS_WHITELIST or any(
            nombre == wl or nombre.startswith(wl + ' ') or nombre.endswith(' ' + wl)
            for wl in PLATOS_WHITELIST
        ):
            resultado.append((nombre, cuenta))
        elif nombre in _cache_es_plato:
            if _cache_es_plato[nombre]:
                resultado.append((nombre, cuenta))
        else:
            por_consultar.append((nombre, cuenta))
    if not por_consultar:
        return resultado
    if not _GEMINI_KEY:
        for nombre, cuenta in por_consultar:
            if PATRONES_NO_PLATO.search(nombre):
                _cache_es_plato[nombre] = False
            else:
                _cache_es_plato[nombre] = True
                resultado.append((nombre, cuenta))
        return resultado
    nombres_consultar = [n for n, _ in por_consultar]
    prompt = (
        f"Clasifica estos candidatos a NOMBRE DE PLATO: {nombres_consultar}\n"
        "es_plato → nombre concreto de un plato, bebida o ingrediente.\n"
        "  Válido: 'tikka masala', 'tarta de queso', 'lomo bajo', 'croquetas de cecina'\n"
        "no_es_plato → TODO lo que contenga: verbos (pedi, probamos, comer, gusta),\n"
        "  adjetivos valorativos (divino, increíble, espectacular, rico, bonito),\n"
        "  descripciones genéricas (restaurante de X, barra de X, solo X, verdad que X).\n"
        "  Rechazar también frases como 'hamburguesa de sushi' si sushi no es ingrediente.\n"
        "REGLA: si contiene verbo o adjetivo valorativo → no_es_plato SIN EXCEPCIÓN.\n"
        'JSON SOLO: {"es_plato":[...],"no_es_plato":[...]}'
    )
    parsed = _gemini_json(prompt, max_tokens=250)
    aprobados = set(parsed.get("es_plato", []))
    rechazados_log = []
    for nombre, cuenta in por_consultar:
        if nombre in aprobados:
            _cache_es_plato[nombre] = True
            resultado.append((nombre, cuenta))
        else:
            _cache_es_plato[nombre] = False
            rechazados_log.append(nombre)
    if not hasattr(filtrar_platos_con_gemini, '_ultimo_rechazo'):
        filtrar_platos_con_gemini._ultimo_rechazo = []
    filtrar_platos_con_gemini._ultimo_rechazo = rechazados_log
    return resultado

ADJETIVOS_INICIO = {
    'buena','bueno','buenos','buenas','gran','grande','grandes',
    'muy','mejor','peor','rica','rico','ricos','ricas',
    'excelente','excelentes','increíble','increíbles','espectacular',
    'fantástico','fantástica','maravilloso','maravillosa','perfecto','perfecta',
    'impecable','excepcional','deliciosa','delicioso','sabroso','sabrosa',
    'calidad','precio','buen','súper','super',
    'servicio','camarero','camarera','barco','lugar','ambiente','trato',
    'mejor','mejores','peores','comida','cocina',
}

DIMENSIONES_KEYWORDS = {
    'servicio':  ['servicio','camarero','camarera','atención','atento','amable',
                  'profesional','trato','pendiente','recomendó','recomendaron'],
    'comida':    ['comida','plato','carta','sabor','calidad','producto','fresco',
                  'rico','riquísimo','delicioso','exquisito'],
    'ambiente':  ['ambiente','local','decoración','acogedor','elegante','bonito',
                  'atmósfera','espacio','tranquilo','íntimo'],
    'precio':    ['precio','caro','barato','relación','calidad-precio','coste','euros'],
    'velocidad': ['rápido','lento','espera','tardó','tiempo','prisas','ágil'],
    'ruido':     ['ruido','ruidoso','silencioso','tranquilo','música'],
    'limpieza':  ['limpio','limpieza','higiene','aseado'],
    # Nuevas dimensiones — conteo estadístico (menciones, pos/neg, avg_stars)
    'ninos':     ['niño','niños','niña','niñas','bebé','bebés','bebe','bebes',
                  'familiar','familia','infantil','sillita','trona','tronas',
                  'peques','pequeños','crío','crios','chaval','chavales'],
    'mascotas':  ['perro','perros','mascota','mascotas','can','canes','peludo',
                  'peludos','admiten perros','perros bienvenidos','dog','pet',
                  'animales','terraza perros'],
    'terraza':   ['terraza','terrazas','exterior','exteriores','al aire libre',
                  'fuera','jardín','jardin','patio','veladores','parasol'],
    'vistas':    ['vistas','vista','panorámica','panoramica','paisaje',
                  'azotea','rooftop','mirador','skyline','horizonte',
                  'vistas al','vistas de','vistas desde'],
    'musica_directo': ['música en directo','música directo','directo','concierto',
                       'actuación','actuacion','banda','grupo en vivo','en vivo',
                       'jazz','flamenco en directo','cantante','show'],
    'romantico': ['romántico','romantico','romántica','romantica','íntimo','intimo',
                  'íntima','intima','pareja','cena romántica','velas','vela',
                  'especial','detalle','sorpresa','aniversario','amor'],
}

# Criterios que tienen clasificación semántica Gemini (además del conteo estadístico)
CRITERIOS_GEMINI = set(CRITERIOS_SIGNAL.keys())  # ninos, mascotas, terraza, vistas, musica_directo, romantico

def clasificar_nombres(candidatos: list) -> set:
    por_consultar = [t for t in candidatos if t not in _cache_nombres]
    if por_consultar:
        prompt = (
            f"Clasifica estas palabras: {por_consultar}\n"
            "nombres_persona = nombres propios de personas\n"
            "no_nombre = todo lo demas (adjetivos, sustantivos, lugares, comida)\n"
            "Si dudas, pon en no_nombre.\n"
            'Responde SOLO con: {"nombres_persona": [...], "no_nombre": [...]}'
        )
        parsed = _gemini_json(prompt, max_tokens=80)
        for t in parsed.get("nombres_persona", []):
            if t in por_consultar: _cache_nombres[t] = True
        for t in parsed.get("no_nombre", []):
            if t in por_consultar: _cache_nombres[t] = False
        for t in por_consultar:
            if t not in _cache_nombres: _cache_nombres[t] = None
    return {t for t in candidatos if _cache_nombres.get(t) is True}

def _dedup_local(platos: list) -> list:
    import unicodedata as _ud

    def _nk(nombre):
        s = nombre.lower().strip()
        s = _ud.normalize('NFD', s)
        s = ''.join(c for c in s if _ud.category(c) != 'Mn')
        s = re.sub(r'^(el |la |los |las |un |una |)', '', s)
        s = re.sub(r'(etas|etes)$', 'eta', s)
        s = re.sub(r'(ones)$', 'on', s)
        s = re.sub(r'([aeiou])s$', r'\1', s)
        s = re.sub(r'([^aeiou])es$', r'\1e', s)
        return ' '.join(sorted(s.split()))

    grupos = {}
    for nombre, cuenta in platos:
        key = _nk(nombre)
        if key in grupos:
            canon, total = grupos[key]
            mejor = nombre if len(nombre) > len(canon) else canon
            grupos[key] = (mejor, total + cuenta)
        else:
            grupos[key] = (nombre, cuenta)

    res = [(n, c) for n, c in grupos.values()]
    res.sort(key=lambda x: -x[1])
    return res


def normalizar_platos(platos: list) -> list:
    if not platos:
        return platos
    vistos: dict = {}
    for nombre, cuenta in platos:
        vistos[nombre] = vistos.get(nombre, 0) + cuenta
    platos = sorted(vistos.items(), key=lambda x: -x[1])
    if not _GEMINI_KEY:
        return platos
    nombres = [t for t, _ in platos]
    cache_key = '|'.join(nombres)
    if cache_key in _cache_normalizacion:
        corregidos = _cache_normalizacion[cache_key]
    else:
        prompt = (
            f"Normaliza estos nombres de platos: {nombres}\n"
            "- Corrige ortografia y acentos (nan->naan, bunuelos->buñuelos)\n"
            "- Fusiona variantes del MISMO plato con el nombre canónico\n"
            "- Minusculas, no traduzcas\n"
            "- Devuelve exactamente el mismo numero de elementos en el mismo orden\n"
            'Responde SOLO con: {"nombres": ["nombre1", "nombre2"]}'
        )
        parsed = _gemini_json(prompt, max_tokens=120)
        corregidos = parsed.get("nombres", [])
        if len(corregidos) == len(platos):
            _cache_normalizacion[cache_key] = corregidos
        else:
            return platos
    if len(corregidos) != len(platos):
        return platos
    fusionados: dict = {}
    for i, nombre_corr in enumerate(corregidos):
        nombre_corr = nombre_corr.lower().strip()
        fusionados[nombre_corr] = fusionados.get(nombre_corr, 0) + platos[i][1]
    return sorted(fusionados.items(), key=lambda x: -x[1])

# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────
def limpiar(texto) -> str:
    if not isinstance(texto, str):
        texto = '' if texto is None or (isinstance(texto, float) and np.isnan(texto)) else str(texto)
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def estrellas_a_categoria(e: int) -> str:
    if e >= 4: return 'positivo'
    if e == 3: return 'neutro'
    return 'negativo'

def analizar_sentimiento_batch(textos):
    textos = [t if isinstance(t, str) and t.strip() else "sin texto" for t in textos]
    resultados = sentiment_pipeline(textos, truncation=True, max_length=512)
    out = []
    for r in resultados:
        e = int(r['label'].split()[0])
        out.append((e, round(r['score'],3), estrellas_a_categoria(e)))
    return out

def porcentaje_menciones(serie, keywords):
    total = len(serie)
    n = sum(1 for r in serie if any(kw in limpiar(str(r)) for kw in keywords))
    return n, round(n/total*100,1)

def _extraer_freq_raw(serie_reviews):
    freq_resenas = Counter()
    for r in serie_reviews:
        texto    = limpiar(str(r))
        palabras = re.findall(r"[a-záéíóúüñ]{3,}", texto)
        vistos   = set()
        for i in range(len(palabras)-1):
            a, b = palabras[i], palabras[i+1]
            if a not in NO_PLATOS and b not in NO_PLATOS:
                bg = f"{a} {b}"
                if bg not in vistos and not PATRONES_NO_PLATO.search(a) and not PATRONES_NO_PLATO.search(b):
                    vistos.add(bg)
        palabras_en_bg = {w for bg in vistos for w in bg.split()}
        for p in palabras:
            if p in PLATOS_WHITELIST and p not in palabras_en_bg and p not in vistos:
                vistos.add(p)
        for t in vistos:
            freq_resenas[t] += 1
    return freq_resenas

def extraer_platos(serie_reviews, n=5, nombre_restaurante=''):
    """
    Extrae nombres de platos de las reseñas.
    Estrategia:
    1. Si una palabra está en PLATOS_WHITELIST → contar directamente.
    2. Bigramas solo si AMBAS palabras son sustantivos de comida
       (ninguna en NO_PLATOS ni ADJETIVOS_INICIO ni verbos).
    3. Whitelist siempre gana sobre bigrama si cubre la misma palabra.
    """
    freq_whitelist = Counter()  # palabras en whitelist
    freq_bigramas  = Counter()  # bigramas limpios
    freq_res_wl    = Counter()  # por reseña (whitelist)
    freq_res_bg    = Counter()  # por reseña (bigramas)

    # Palabras que no pueden formar parte de ningún nombre de plato
    _NUNCA = NO_PLATOS | STOP_ES | set(ADJETIVOS_INICIO) | {
        'pedi','pedimos','probe','probamos','tome','tomamos','cenamos','comimos',
        'recomiendo','recomendamos','trajeron','pusieron','traen','ponen',
        'hacen','sirven','tienen','pedido','probado','tomado','cenado',
        'divino','divina','genial','increible','espectacular','agradable',
        'bonito','bonita','solo','sólo','verdad','gusta','encantan','encanta',
        'restaurante','local','sitio','barra','rueda','marca','ración','tapa',
        'rico','rica','bueno','buena','malo','mala','mejor','peor',
    }

    for resena in serie_reviews:
        texto    = limpiar(str(resena))
        palabras = re.findall(r'[a-záéíóúüñ]+', texto)
        vis_wl   = set()
        vis_bg   = set()

        for i, p in enumerate(palabras):
            # 1. Whitelist directa
            if p in PLATOS_WHITELIST:
                freq_whitelist[p] += 1
                if p not in vis_wl:
                    freq_res_wl[p] += 1
                    vis_wl.add(p)

            # 2. Bigramas: la palabra actual inicia bigrama solo si
            #    no está en _NUNCA y la siguiente tampoco
            if i < len(palabras) - 1:
                a, b = palabras[i], palabras[i+1]
                if (a not in _NUNCA and b not in _NUNCA
                        and len(a) >= 3 and len(b) >= 3):
                    bg = f'{a} {b}'
                    freq_bigramas[bg] += 1
                    if bg not in vis_bg:
                        freq_res_bg[bg] += 1
                        vis_bg.add(bg)

            # 3. Trigramas solo si palabra central es "de/a/al/con"
            if i < len(palabras) - 2:
                a, prep, c = palabras[i], palabras[i+1], palabras[i+2]
                if (prep in ('de','a','al','con','en','y')
                        and a not in _NUNCA and c not in _NUNCA
                        and len(a) >= 3 and len(c) >= 3):
                    tg = f'{a} {prep} {c}'
                    freq_bigramas[tg] += 1
                    if tg not in vis_bg:
                        freq_res_bg[tg] += 1
                        vis_bg.add(tg)

    # Construir candidatos: combinar whitelist + bigramas
    # Whitelist gana sobre bigrama que la contiene
    palabras_en_bg = {w for bg in freq_bigramas for w in bg.split()}
    candidatos = []

    # Añadir whitelist si no está cubierta por bigrama mejor
    for p, cnt in freq_res_wl.items():
        if cnt >= 2 or (cnt >= 1 and p in PLATOS_WHITELIST):
            # Solo añadir si no hay bigrama con más menciones que la contiene
            mejor_bg = max(
                (freq_res_bg.get(bg, 0) for bg in freq_bigramas if p in bg.split()),
                default=0
            )
            if mejor_bg <= cnt:
                candidatos.append((p, cnt))

    # Añadir bigramas/trigramas con >= 2 menciones en reseñas distintas
    for bg, cnt in freq_res_bg.items():
        if cnt >= 2:
            candidatos.append((bg, cnt))

    # Eliminar duplicados (si bigrama ya cubre la whitelist word)
    wl_cubiertas = {w for bg,_ in candidatos if ' ' in bg for w in bg.split()
                    if w in PLATOS_WHITELIST}
    candidatos = [(t, c) for t, c in candidatos
                  if ' ' in t or t not in wl_cubiertas]

    candidatos.sort(key=lambda x: -x[1])

    # Blacklist y filtro de platos válidos antes de Gemini
    def _plato_valido(t):
        return (t not in PLATOS_BLACKLIST
                and not any(p in _NUNCA for p in t.lower().split()))

    platos_sin_filtrar = []
    for t, c in candidatos:
        if not _plato_valido(t):
            continue
        if PATRONES_NO_PLATO.search(t):
            continue
        platos_sin_filtrar.append((t, c))
        if len(platos_sin_filtrar) >= n * 3:
            break
    platos_filtrados = filtrar_platos_con_gemini(platos_sin_filtrar, nombre_restaurante)
    platos_filtrados.sort(key=lambda x: -x[1])
    platos = platos_filtrados[:n]
    if MODO_ECONOMICO:
        platos_norm = platos  # saltar normalización para ahorrar tokens
    else:
        platos_norm = normalizar_platos(platos)
        platos_norm.sort(key=lambda x: -x[1])
        platos_norm = platos_norm[:n]
    if not platos_norm:
        motivo = "ningún candidato superó >= 2 menciones" if not candidatos_2 else \
                 f"{len(candidatos_2)} candidatos rechazados por filtros/Gemini"
        print(f"  ⚠️  [{nombre_restaurante}] 0 platos detectados — {motivo}")
        rechazados = getattr(filtrar_platos_con_gemini, '_ultimo_rechazo', [])
        if rechazados:
            print(f"       Rechazados: {rechazados}")
    return platos_norm

def extraer_personal(serie_reviews, platos_set, n_resenas, n=5):
    freq_total   = Counter()
    freq_resenas = Counter()
    for r in serie_reviews:
        texto    = limpiar(str(r))
        palabras = re.findall(r"[a-záéíóúüñ]{4,}", texto)
        tokens_set = set()
        for t in palabras:
            if t not in STOP_ES and t not in NO_PLATOS and t not in platos_set:
                freq_total[t] += 1
                tokens_set.add(t)
        for t in tokens_set:
            freq_resenas[t] += 1
    candidatos = [
        (t, freq_total[t]) for t in freq_total
        if freq_resenas.get(t,0) >= 2
        and freq_resenas.get(t,0) <= n_resenas * 0.40
    ]
    candidatos.sort(key=lambda x: -x[1])
    nombres = clasificar_nombres([t for t,_ in candidatos[:30]])
    personal = [(t,c) for t,c in candidatos if t in nombres]
    return personal[:n]

# ─────────────────────────────────────────────
# ANÁLISIS POR RESTAURANTE
# ─────────────────────────────────────────────
def analizar_restaurante(df_r):
    nombre     = df_r['Restaurante'].iloc[0]
    rid        = int(df_r['Id_Restaurante'].iloc[0])
    direccion  = df_r['Direccion'].iloc[0] if 'Direccion' in df_r.columns else ''
    valoracion = df_r['Valoracion'].mean()
    n          = len(df_r)

    # Sentimiento
    textos = df_r['Review'].astype(str).tolist()
    sent_res = []
    for i in range(0, len(textos), BATCH_SIZE):
        sent_res += analizar_sentimiento_batch(textos[i:i+BATCH_SIZE])

    df_r = df_r.copy()
    df_r['estrellas']     = [x[0] for x in sent_res]
    df_r['score_modelo']  = [x[1] for x in sent_res]
    df_r['sentimiento']   = [x[2] for x in sent_res]
    df_r['baja_confianza']= df_r['score_modelo'] < MIN_CONFIANZA

    sent_global    = Counter(df_r['sentimiento'])
    estrellas_dist = Counter(df_r['estrellas'])
    avg_stars      = round(df_r['estrellas'].mean(), 3)
    total_sent     = sum(sent_global.values())
    pct_pos = round(sent_global.get('positivo',0)/total_sent*100,1)
    pct_neg = round(sent_global.get('negativo',0)/total_sent*100,1)
    pct_neu = round(sent_global.get('neutro',0)/total_sent*100,1)
    n_baja  = int(df_r['baja_confianza'].sum())

    # Dimensiones — conteo estadístico (menciones, pos/neg, avg_stars)
    dim_data = {}
    for dim, kws in DIMENSIONES_KEYWORDS.items():
        menciones = sum(1 for r in df_r['Review'] if any(kw in limpiar(str(r)) for kw in kws))
        subset    = df_r[df_r['Review'].apply(lambda x: any(kw in limpiar(str(x)) for kw in kws))]
        sc        = Counter(subset['sentimiento'])
        avg_e     = round(subset['estrellas'].mean(),2) if len(subset) else None
        dim_data[dim] = {
            'menciones': menciones,
            'pct':       round(menciones/n*100,1),
            'positivo':  sc.get('positivo',0),
            'negativo':  sc.get('negativo',0),
            'neutro':    sc.get('neutro',0),
            'avg_stars': avg_e,
            'rep':       menciones >= MIN_MENCIONES,
        }

    # Criterios cualitativos — clasificación semántica Gemini sobre fragmentos
    criterios, criterios_frases = clasificar_criterios_gemini(rid, nombre, df_r['Review'].tolist())
    criterios_log = ', '.join(f'{k}={"✓" if v else "✗"}' for k,v in criterios.items())
    print(f"    criterios: {criterios_log}")

    # Servicio destacado — extraer nombres del personal de las reseñas
    # Se hace siempre (también en MODO_ECONOMICO) porque reutiliza clasificar_nombres
    # que ya tiene caché y no cuesta tokens para nombres ya conocidos
    if not MODO_ECONOMICO:
        personal    = extraer_personal(df_r['Review'], platos_set, n)
        personal_str_nuevo = ', '.join([f"{t.capitalize()}({c})" for t,c in personal])
    else:
        personal_str_nuevo = None  # se asigna abajo desde CSV previo

    # Platos
    # Una sola llamada a Gemini (n=15) — top5 es simplemente los primeros 5
    platos_todos = extraer_platos(df_r['Review'], n=15, nombre_restaurante=nombre)
    platos_todos = _dedup_local(platos_todos)[:15]  # fusionar variantes localmente

    # ── Recuperación de platos de whitelist con alta frecuencia no detectados ──
    # Segundo barrido: buscar platos de PLATOS_WHITELIST que aparezcan en >= 3
    # reseñas distintas pero que el extractor de bigramas haya perdido
    # (ej: "cocido madrileño" porque "madrileño" está en NO_PLATOS)
    MIN_MENCIONES_RECUPERACION = 3
    platos_ya_detectados = {limpiar(p) for p, _ in platos_todos}
    textos_resenas = df_r['Review'].astype(str).tolist()
    for plato_wl in sorted(PLATOS_WHITELIST):
        plato_limpio = limpiar(plato_wl)
        # Saltar si ya está detectado o es muy genérico (< 4 chars)
        if len(plato_limpio) < 4:
            continue
        if any(plato_limpio in det or det in plato_limpio
               for det in platos_ya_detectados):
            continue
        # Contar menciones en reseñas distintas
        menciones = sum(1 for r in textos_resenas if plato_limpio in limpiar(str(r)))
        if menciones >= MIN_MENCIONES_RECUPERACION:
            platos_todos.append((plato_wl, menciones))
            platos_ya_detectados.add(plato_limpio)
    # Re-ordenar por menciones
    platos_todos.sort(key=lambda x: -x[1])
    platos_todos = platos_todos[:15]

    platos_top   = platos_todos[:5]
    platos_str   = ', '.join([f"{p}({c})" for p,c in platos_top])
    todos_str    = ', '.join([f"{p}({c})" for p,c in platos_todos])
    platos_set   = {p for p,_ in platos_todos}

    # Personal — en modo económico reusar del CSV previo
    if MODO_ECONOMICO:
        personal_str = ''
        if os.path.exists(OUTPUT_CSV):
            try:
                _prev = pd.read_csv(OUTPUT_CSV)
                _fila_prev = _prev[_prev['id_restaurante'] == rid]
                if not _fila_prev.empty and 'personal_destacado' in _prev.columns:
                    personal_str = str(_fila_prev['personal_destacado'].iloc[0])
                    if personal_str == 'nan': personal_str = ''
            except Exception:
                pass
    else:
        personal_str = personal_str_nuevo or ''

    # Frases de servicio — extraer fragmentos positivos de reseñas que mencionan
    # al personal por nombre o elogian el servicio (sin coste Gemini)
    servicio_frases = ''
    kws_servicio = DIMENSIONES_KEYWORDS.get('servicio', [])
    frags_serv = _extraer_fragmentos(df_r['Review'].tolist(), kws_servicio, ventana=50)
    frags_serv_pos = [f for f in frags_serv
                      if not any(_fragmento_tiene_negacion(f, kw) for kw in kws_servicio)]
    if frags_serv_pos:
        if _GEMINI_KEY:
            servicio_frases = _parafrasear_frases(frags_serv_pos[:3], f"el servicio de {nombre}")
        else:
            servicio_frases = ' | '.join(frags_serv_pos[:3])

    # Reseñas destacadas — parafraseadas por Gemini (privacidad + calidad)
    resenas_destacadas = _generar_resenas_destacadas(df_r, nombre)

    # TF-IDF
    textos_limpios = df_r['Review'].apply(limpiar).tolist()
    try:
        vec = TfidfVectorizer(stop_words=list(STOP_ES), min_df=2,
                              max_features=50, ngram_range=(1,2))
        X   = vec.fit_transform(textos_limpios)
        mt  = np.asarray(X.mean(axis=0)).flatten()
        ti  = mt.argsort()[::-1][:5]
        tfidf_str = ', '.join(vec.get_feature_names_out()[ti])
    except Exception:
        tfidf_str = ''

    fila = {
        'id_restaurante':       rid,
        'nombre':               nombre,
        'direccion':            direccion,
        'n_resenas':            n,
        'valoracion_google':    round(valoracion,2),
        'avg_estrellas_modelo': avg_stars,
        'pct_positivo':         pct_pos,
        'pct_neutro':           pct_neu,
        'pct_negativo':         pct_neg,
        'n_resenas_negativas':  sent_global.get('negativo',0),
        'n_baja_confianza':     n_baja,
        'estrellas_5':          estrellas_dist.get(5,0),
        'estrellas_4':          estrellas_dist.get(4,0),
        'estrellas_3':          estrellas_dist.get(3,0),
        'estrellas_2':          estrellas_dist.get(2,0),
        'estrellas_1':          estrellas_dist.get(1,0),
        **{f'{d}_menciones': dim_data[d]['menciones'] for d in DIMENSIONES_KEYWORDS},
        **{f'{d}_pct':        dim_data[d]['pct']       for d in DIMENSIONES_KEYWORDS},
        **{f'{d}_pos':        dim_data[d]['positivo']  for d in DIMENSIONES_KEYWORDS},
        **{f'{d}_neg':        dim_data[d]['negativo']  for d in DIMENSIONES_KEYWORDS},
        **{f'{d}_avg_stars':  dim_data[d]['avg_stars'] for d in DIMENSIONES_KEYWORDS},
        **{f'{d}_rep':        dim_data[d]['rep']       for d in DIMENSIONES_KEYWORDS},
        # Criterios cualitativos — booleanos validados por Gemini
        **{f'criterio_{c}': criterios[c] for c in CRITERIOS_SIGNAL},
        # Frases de reseñas que justifican cada criterio (coste cero)
        **{f'criterio_{c}_frases': criterios_frases.get(c,'') for c in CRITERIOS_SIGNAL},
        # Frases de servicio destacado
        'servicio_frases':      servicio_frases,
        'resenas_destacadas':   resenas_destacadas,
        'top5_platos':          platos_str,
        'todos_platos':         todos_str,
        'personal_destacado':   personal_str,
        'terminos_tfidf':       tfidf_str,
        'cocina_detectada':     _detectar_cocina_restaurante(todos_str, nombre),
    }

    df_r_out = df_r[['Id_Restaurante','Restaurante','Id_review','Review',
                     'Valoracion','estrellas','score_modelo',
                     'sentimiento','baja_confianza']].copy()
    return fila, df_r_out

# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────
print("="*60)
print("Cargando modelo nlptown...")
print("="*60)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=MODELO_HF,
    truncation=True, max_length=512, batch_size=BATCH_SIZE,
)
print("Modelo listo.\n")

if _GEMINI_KEY:
    if MODO_ECONOMICO:
        print(f"✔ Gemini configurado: {_GEMINI_MODEL}")
        print("  Modo económico ACTIVADO — normalización y personal desactivados.")
        print("  Llamadas Gemini por restaurante: ~2 (clasificar platos + criterios)")
        print("  Estimación: €0.01-0.02 por restaurante · €1.25-2.50 para 125 rest.")
    else:
        print(f"✔ Gemini configurado: {_GEMINI_MODEL} (completo: platos + normalización + nombres + criterios)")
        print("  Estimación: €0.03-0.05 por restaurante · €3.75-6.25 para 125 rest.")
else:
    print("⚠️  Sin GEMINI_API_KEY — nombres, normalización y criterios en modo heurístico")
print()

# ─────────────────────────────────────────────
# CARGA Y LIMPIEZA
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
n_antes = len(df)
df['Review'] = df['Review'].astype(str).str.strip()
df = df[df['Review'].str.lower() != 'nan']
df = df[df['Review'] != '']
print(f"Reseñas cargadas   : {n_antes}")
print(f"Reseñas eliminadas : {n_antes - len(df)}")
print(f"Reseñas válidas    : {len(df)}\n")

# ── Enriquecer con direcciones del ranking ────────────────────────────────────
RANKING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ranking.csv')
if os.path.exists(RANKING_PATH):
    df_ranking = pd.read_csv(RANKING_PATH, sep=';', skiprows=1)
    df_ranking = df_ranking[['Id_Restaurante','Dirección']].dropna(subset=['Dirección'])
    df_ranking = df_ranking.rename(columns={'Dirección': 'Direccion'})
    df = df.merge(df_ranking, on='Id_Restaurante', how='left')
    df['Direccion'] = df['Direccion'].fillna('')
    n_con_dir = (df.groupby('Id_Restaurante')['Direccion'].first() != '').sum()
    print(f'Direcciones cargadas de ranking.csv: {n_con_dir} restaurantes\n')
else:
    df['Direccion'] = ''
    print('⚠️  ranking.csv no encontrado — direcciones vacías\n')

ids_todos = sorted(df['Id_Restaurante'].unique())

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
ids_procesados    = set()
ids_incompletos   = set()
filas_resumen     = []
filas_resenas     = []

if os.path.exists(OUTPUT_CSV):
    df_prev = pd.read_csv(OUTPUT_CSV)

    # Detectar columnas nuevas — si faltan dimensiones O criterios, reprocesar todo
    cols_esperadas = (
        [f'{d}_menciones' for d in DIMENSIONES_KEYWORDS] +
        [f'criterio_{c}' for c in CRITERIOS_SIGNAL]
    )
    cols_faltantes = [c for c in cols_esperadas if c not in df_prev.columns]
    if cols_faltantes:
        print(f"⚠️  Columnas nuevas detectadas: {cols_faltantes}")
        print("   Forzando reprocesado completo.\n")
        ids_procesados = set()
        ids_completar  = set()
        filas_resumen  = []
    else:
        def _contar_platos(val):
            if pd.isna(val) or str(val).strip() == '':
                return 0
            return len(re.findall(r'\(\d+\)', str(val)))

        _patron_falso = re.compile(
            r'\b(comida|cocina)\s+(muy\s+)?(rica?|buena?|excelente|deliciosa?|'
            r'riquísima?|increíble|espectacular|fantástica?|perfecta?|'
            r'exquisita?|abundante|fresca?|caliente|fría?|casera?|tradicional)\b',
            re.IGNORECASE
        )
        def _tiene_platos_falsos(val):
            return bool(_patron_falso.search(str(val))) if not pd.isna(val) else False

        df_prev['_n_platos']      = df_prev['top5_platos'].apply(_contar_platos)
        df_prev['_platos_falsos'] = df_prev['top5_platos'].apply(_tiene_platos_falsos)

        ids_completos = set(df_prev[
            (df_prev['_n_platos'] >= 5) & (~df_prev['_platos_falsos'])
        ]['id_restaurante'].astype(int))

        ids_completar = set(df_prev[
            (df_prev['_n_platos'] >= 1) &
            (df_prev['_n_platos'] <  5) &
            (~df_prev['_platos_falsos'])
        ]['id_restaurante'].astype(int))

        ids_incompletos = set(df_prev['id_restaurante'].astype(int)) - ids_completos - ids_completar
        ids_procesados  = ids_completos

        filas_resumen = df_prev[df_prev['id_restaurante'].astype(int).isin(ids_completos)] \
                            .drop(columns=['_n_platos','_platos_falsos'], errors='ignore').to_dict('records')
        print(f"CSV existente: {len(ids_completos)} completos · "
              f"{len(ids_completar)} a completar (sin BERT) · "
              f"{len(ids_incompletos)} a reprocesar completo")
else:
    ids_completar = set()
    print("No existe CSV previo — procesando todo desde cero.")

if os.path.exists(OUTPUT_RESENAS):
    df_res_prev = pd.read_csv(OUTPUT_RESENAS)
    if ids_procesados:
        filas_resenas = [df_res_prev[df_res_prev['Id_Restaurante'].astype(int).isin(ids_procesados)]]

# ── Completar restaurantes con 1-4 platos SIN reprocesar BERT ────────────────
if ids_completar:
    print(f"\nCompletando {len(ids_completar)} restaurantes con 1-4 platos...")
    df_completar = df_prev[df_prev['id_restaurante'].astype(int).isin(ids_completar)].copy()
    df_resenas_prev = pd.read_csv(OUTPUT_RESENAS) if os.path.exists(OUTPUT_RESENAS) else None

    completados = 0
    for _, row in df_completar.iterrows():
        rid        = int(row['id_restaurante'])
        nombre_r   = row['nombre']
        top5_actual = row['top5_platos'] if not pd.isna(row.get('top5_platos','')) else ""

        resenas_r = df[df['Id_Restaurante'] == rid]['Review'].tolist() if df_resenas_prev is None \
                    else df_resenas_prev[df_resenas_prev['Id_Restaurante'] == rid]['Review'].tolist()
        if not resenas_r:
            filas_resumen.append(row.drop(['_n_platos','_platos_falsos'], errors='ignore').to_dict())
            continue

        platos_actuales = []
        if top5_actual:
            for p in top5_actual.split(','):
                p = p.strip()
                if re.search(r'\(\d+\)', p):
                    platos_actuales.append(p)
        nombres_actuales = {re.sub(r'\(\d+\)$','',p).strip().lower() for p in platos_actuales}
        faltan = 5 - len(platos_actuales)

        freq = _extraer_freq_raw(resenas_r)
        candidatos = [(t,c) for t,c in freq.items()
                      if c >= 2 and t.lower() not in nombres_actuales]
        candidatos.sort(key=lambda x: -x[1])
        nuevos_f1 = candidatos[:faltan * 3]

        nuevos_f1_filt = filtrar_platos_con_gemini(nuevos_f1, nombre_r)
        nuevos_f1_filt.sort(key=lambda x: -x[1])
        nuevos_f1_filt = nuevos_f1_filt[:faltan]

        top5_nuevo = platos_actuales + [f"{p}({c})" for p,c in nuevos_f1_filt]

        faltan2 = 5 - len(top5_nuevo)
        if faltan2 > 0:
            nombres_nuevo = {re.sub(r'\(\d+\)$','',p).strip().lower() for p in top5_nuevo}
            mas = [(t,c) for t,c in freq.items()
                   if c >= 1 and t.lower() not in nombres_nuevo
                   and t not in [x for x,_ in nuevos_f1_filt]]
            mas.sort(key=lambda x: -x[1])
            mas_filt = filtrar_platos_con_gemini(mas[:faltan2*4], nombre_r)
            mas_filt.sort(key=lambda x: -x[1])
            top5_nuevo += [f"{p}({c})" for p,c in mas_filt[:faltan2]]

        if top5_nuevo:
            if MODO_ECONOMICO:
                top5_str = ", ".join(top5_nuevo[:5])
            else:
                platos_parsed = [(re.sub(r'\(\d+\)$','',p).strip(),
                                  int(re.search(r'\((\d+)\)$',p).group(1))) for p in top5_nuevo]
                normalizados = normalizar_platos(platos_parsed)
                normalizados.sort(key=lambda x: -x[1])
                top5_str = ", ".join(f"{p}({c})" for p,c in normalizados[:5])
        else:
            top5_str = top5_actual

        fila_dict = row.drop(['_n_platos','_platos_falsos'], errors='ignore').to_dict()
        fila_dict['top5_platos'] = top5_str
        filas_resumen.append(fila_dict)

        n_nuevo = len(re.findall(r'\(\d+\)', top5_str))
        print(f"  [{rid}] {nombre_r}: {len(platos_actuales)}→{n_nuevo} platos | {top5_str[:70]}")
        completados += 1

    _guardar_cache_platos()
    print(f"  {completados} restaurantes completados.\n")

ids_pendientes = [i for i in ids_todos if int(i) not in ids_procesados and int(i) not in ids_completar]
print(f"Total: {len(ids_todos)} | Completos: {len(ids_procesados)} | "
      f"A completar sin BERT: {len(ids_completar)} | "
      f"Pendientes BERT: {len(ids_pendientes)}\n")

_cargar_cache_platos()

# ─────────────────────────────────────────────
# PROCESADO
# ─────────────────────────────────────────────
for i, rid in enumerate(ids_pendientes, 1):
    df_r    = df[df['Id_Restaurante'] == rid]
    nombre_r= df_r['Restaurante'].iloc[0]
    print(f"[{i}/{len(ids_pendientes)}] Restaurante {rid}: {nombre_r} ({len(df_r)} reseñas)...")

    try:
        fila, df_resenas = analizar_restaurante(df_r)
        filas_resumen.append(fila)
        filas_resenas.append(df_resenas)

        pd.DataFrame(filas_resumen).sort_values('id_restaurante').to_csv(OUTPUT_CSV, index=False)
        pd.concat(filas_resenas).sort_values('Id_Restaurante').to_csv(OUTPUT_RESENAS, index=False)
        _guardar_cache_platos()

        print(f"    ✔ {fila['pct_positivo']}% pos | ★{fila['avg_estrellas_modelo']} | {fila['top5_platos'][:70]}")
        print(f"    💶 Coste acumulado estimado: €{coste_estimado():.4f} ({_tok_input//1000}k input / {_tok_output//1000}k output tokens)")

    except Exception as e:
        import traceback
        print(f"    ✗ Error: {e}")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESADO: recuperación de platos y detección de cocina para TODOS
# Se aplica sobre el CSV final ya generado, sin reprocesar BERT
# ─────────────────────────────────────────────────────────────────────────────
print("\nPost-procesando: recuperación de platos whitelist y cocina_detectada...")

df_final = pd.read_csv(OUTPUT_CSV)
df_resenas_cargadas = pd.read_csv(OUTPUT_RESENAS) if os.path.exists(OUTPUT_RESENAS) else df.rename(columns={'Review':'Review'})

MIN_MENCIONES_RECUPERACION = 3
actualizados_platos = 0
actualizados_cocina = 0

for idx, row in df_final.iterrows():
    rid = int(row['id_restaurante'])
    nombre_r = str(row['nombre'])

    # Reseñas de este restaurante
    try:
        resenas_r = df_resenas_cargadas[df_resenas_cargadas['Id_Restaurante'] == rid]['Review'].astype(str).tolist()
    except Exception:
        resenas_r = df[df['Id_Restaurante'] == rid]['Review'].astype(str).tolist()

    if not resenas_r:
        continue

    # Platos ya detectados
    todos_str = str(row.get('todos_platos', '') or '')
    platos_actuales = {}
    for parte in todos_str.split(','):
        m = re.match(r'^(.+?)\((\d+)\)$', parte.strip())
        if m:
            platos_actuales[limpiar(m.group(1).strip())] = int(m.group(2))

    # Segundo barrido: buscar platos de whitelist no detectados
    nuevos = []
    for plato_wl in sorted(PLATOS_WHITELIST):
        plato_limpio = limpiar(plato_wl)
        if len(plato_limpio) < 4:
            continue
        # Ya está detectado (exact o parcial)
        if any(plato_limpio in det or det in plato_limpio
               for det in platos_actuales):
            continue
        menciones = sum(1 for r in resenas_r if plato_limpio in limpiar(str(r)))
        if menciones >= MIN_MENCIONES_RECUPERACION:
            nuevos.append((plato_wl, menciones))
            platos_actuales[plato_limpio] = menciones  # evitar duplicados

    if nuevos:
        nuevos.sort(key=lambda x: -x[1])
        nuevos_str = ', '.join(f"{p}({c})" for p, c in nuevos)
        if todos_str and todos_str != 'nan':
            df_final.at[idx, 'todos_platos'] = nuevos_str + ', ' + todos_str
        else:
            df_final.at[idx, 'todos_platos'] = nuevos_str
        todos_str = df_final.at[idx, 'todos_platos']
        actualizados_platos += 1
        print(f"  [{rid}] {nombre_r}: +{[f'{p}({c})' for p,c in nuevos]}")

    # Detectar cocina
    cocina = _detectar_cocina_restaurante(todos_str, nombre_r)
    current = str(row.get('cocina_detectada', '') or '')
    if cocina and current in ('', 'nan'):
        df_final.at[idx, 'cocina_detectada'] = cocina
        actualizados_cocina += 1
    elif not cocina and current in ('', 'nan'):
        df_final.at[idx, 'cocina_detectada'] = ''

    # personal_destacado: aplicar si está vacío
    personal_actual = str(row.get('personal_destacado', '') or '')
    if personal_actual in ('', 'nan'):
        personal = extraer_personal(resenas_r, platos_set=set(platos_actuales.keys()), n_resenas=len(resenas_r), n=3)
        if personal:
            df_final.at[idx, 'personal_destacado'] = ', '.join(
                f"{n.capitalize()}({c})" for n, c in personal
            )

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"  Platos recuperados en {actualizados_platos} restaurantes")
print(f"  Cocina detectada en {actualizados_cocina} restaurantes")
print(f"  CSV actualizado: {OUTPUT_CSV}")

print(f"\n{'='*60}\n¡Completado!\n  → {OUTPUT_CSV}\n  → {OUTPUT_RESENAS}")
