"""
Análisis masivo de reseñas — 125 restaurantes
Sentimiento: nlptown (local)
Platos: tokens + bigramas con filtro NO_PLATOS
Nombres: Gemini (solo para clasificar nombres de personas)
Normalización: Gemini (solo para corregir ortografía de platos)
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
    # Sustantivos genéricos de restaurante
    'restaurante','comida','cocina','servicio','trato','atención','experiencia','lugar',
    'sitio','personal','camarero','camarera','ambiente','precio','calidad','carta',
    'mesa','plato','platos','producto','productos','cocina','local','establecimiento',
    # Adjetivos de valoración (con y sin tilde)
    'buena','bueno','buen','buenos','buenas','rico','rica','ricos','ricas',
    'riquísimo','delicioso','deliciosa','deliciosos','deliciosas',
    'exquisito','exquisita','espectacular','increíble','perfecto','perfecta',
    'impecable','excepcional','excelente','fantástico','fantástica',
    'maravilloso','maravillosa','inmejorable','estupendo','estupenda',
    'sobresaliente','alucinante','brutal','genial','fenomenal',
    'recomendable','insuperable','único','única','especial','auténtico',
    'fresco','fresca','sabroso','sabrosa','increible','magnifico',
    'deliciosa','mejor','peor','igual','mismo','misma',
    # Adjetivos sin tilde (versiones normalizadas)
    'rapido','rapida','amable','amables','atento','atenta','atentos','atentas',
    'agradable','agradables','simpatico','simpatica','profesional','profesionales',
    'eficiente','eficientes','educado','educada','cercano','cercana',
    # Adverbios y valoraciones
    'súper','super','muy','mucho','mucha','muchos','muchas','poco','poca',
    'siempre','nunca','también','además','realmente','totalmente',
    'absolutamente','simplemente','especialmente','principalmente',
    'gracias','encantado','encantada','satisfecho','satisfecha',
    # Verbos frecuentes
    'volver','volveremos','volvería','repetiremos','repetir',
    'recomiendo','recomendamos','recomendable','recomendado','recomendada',
    'pedimos','pedí','tomamos','probamos','atendió','atendieron',
    'encantó','gustó','quedamos','salimos','fuimos','vinimos',
    # Sustantivos de experiencia
    'visita','primera','segunda','tercera','última','próxima',
    'ocasión','celebración','cumpleaños','aniversario','cena','almuerzo',
    'reserva','mesa','cuenta','nota','propina',
    # Términos detectados en producción que no son platos
    'acogedor','acogedora','principal','destacar','destacado','destacada',
    'verdad','variedad','variedad de','tiempo','chicos','chicas',
    'comer','tomar','pedir','puse','puse','salir','venir','hacer',
    'gente','personas','novio','novia','marido','mujer','pareja',
    'brunch','brunch de','momento','momentos','sabor','sabores',
    'cantidad','cantidades','racion','raciones','ración','raciones',
    'presentacion','presentación','vista','vistas',
    # Ciudades y lugares
    'madrid','barcelona','españa','galicia','india','italia','japón',
    'mexico','peru','venezuela','colombia','cuba','argentina',
    # Otros muy frecuentes que no son platos
    'familia','amigos','pareja','grupo','equipo','compañeros',
    'menú','degustación','carta','precios','euros','coste',
    'duda','acierto','momento','punto','nivel','tipo','clase',
    'detalle','detalles','nada','algo',
    # Bigramas genéricos frecuentes (adjetivo+sustantivo)
    'sin duda','por supuesto','desde luego','muy recomendable',
    'plato principal','gran plato','buen plato',
    'calidad precio','buena relacion',
    # "comida X" genéricos — la cocina/cuisine, no un plato concreto
    'comida india','comida italiana','comida japonesa','comida china',
    'comida mexicana','comida peruana','comida venezolana','comida colombiana',
    'comida española','comida francesa','comida griega','comida árabe',
    'comida arabe','comida turca','comida americana','comida internacional',
    'comida casera','comida tradicional','comida fusion','comida fusión',
    'comida rapida','comida rápida','comida callejera',
    'cocina india','cocina italiana','cocina japonesa','cocina china',
    'cocina mexicana','cocina peruana','cocina venezolana','cocina española',
    'cocina francesa','cocina mediterránea','cocina mediterranea',
    'cocina tradicional','cocina fusion','cocina fusión','cocina casera',
}

# ── Palabras que SÓLO son platos/comidas (pasan el filtro siempre) ──────────
# Añade aquí los platos recurrentes de tu dataset para blindarlos
PLATOS_WHITELIST = {
    # Carnes y pescados
    'pulpo','chuletón','chuleton','cochinillo','lechazo','cordero','rabo','carrillera',
    'bacalao','merluza','rodaballo','lubina','dorada','rape','salmón','salmon',
    'atún','atun','bonito','anchoas','gambas','langostinos','calamares','chipirones',
    'mejillones','almejas','berberechos','navajas','percebes','langosta','bogavante',
    'sepia','cocochas',
    # Ibérico y embutidos
    'jamón','jamon','lomo','chorizo','morcilla','salchichón','salchicon','sobrasada',
    'cecina','foie','mollejas',
    # Arroces y pasta
    'paella','arroz','risotto','fideuá','fideua','pasta','espaguetis','lasaña','lasana',
    'carbonara','penne','ravioli','tagliatelle',
    # Tapas y entrantes
    'croquetas','croqueta','patatas bravas','tortilla','gazpacho','salmorejo',
    'ensaladilla','berenjenas','pimientos','pisto','ratatouille','hummus',
    'guacamole','nachos','bruschetta','focaccia','tostas',
    # Platos internacionales (india, venezolana, etc.)
    'tikka masala','biryani','naan','samosa','curry','korma','dal','tandoori',
    'arepas','arepa','pabellón','pabellon','cachapas','tequeños','tequenos',
    'empanadas','empanada','ceviche','tiradito','lomo saltado','causa','anticuchos',
    'sushi','sashimi','ramen','gyozas','edamame','tempura','miso',
    'tacos','burrito','quesadilla','fajitas',
    # Postres
    'tarta','tartaleta','coulant','brownie','tiramisú','tiramisu','panna cotta',
    'flan','crema catalana','helado','mousse','cheesecake','mochi',
    'churros','soufflé','souffle','crepe','waffle','suspiro',
    # Bebidas
    'sangría','sangria','mojito','margarita','caipirinha','negroni','martini',
    # Buñuelos y similares (del dataset)
    'buñuelos','bunuelos','buñuelos de bacalao',
    # Mariscos compuestos
    'paella de mariscos','paella de marisco',
}

# Patrón para detectar que un token NO puede ser un plato:
# - empieza con prefijo gramatical
# - termina en sufijo de adjetivo/adverbio típico
PATRONES_NO_PLATO = re.compile(
    r'^(muy|gran|super|súper|bien|mal|sin|con|para|como|desde|hasta|'
    r'nuestro|nuestra|vuestro|vuestra|todo|toda|todos|todas|'
    r'primer|primera|segundo|segunda|otro|otra|cada|mismo|misma)$'
    r'|'
    r'(mente|ísimo|ísima|ísimos|ísimas|ción|sión)$'
)

# ─────────────────────────────────────────────────────────────────────────────
# Cache persistente para validación Gemini de platos
# Se guarda en disco para no repetir llamadas entre ejecuciones
_CACHE_PLATOS_PATH = os.path.join(OUTPUT_DIR, ".cache_platos_gemini.json")
_cache_es_plato: dict = {}

def _cargar_cache_platos():
    global _cache_es_plato
    if os.path.exists(_CACHE_PLATOS_PATH):
        try:
            with open(_CACHE_PLATOS_PATH, 'r', encoding='utf-8') as f:
                _cache_es_plato = _json.load(f)
            print(f"Cache Gemini cargado: {len(_cache_es_plato)} términos conocidos.")
        except Exception:
            _cache_es_plato = {}

def _guardar_cache_platos():
    try:
        with open(_CACHE_PLATOS_PATH, 'w', encoding='utf-8') as f:
            _json.dump(_cache_es_plato, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  No se pudo guardar cache Gemini: {e}")

def filtrar_platos_con_gemini(candidatos: list, nombre_restaurante: str = "") -> list:
    """
    Usa Gemini para decidir cuáles candidatos son platos/comidas reales.
    Devuelve solo los que superan el filtro.
    Si no hay GEMINI_API_KEY, aplica solo heurísticas.
    """
    if not candidatos:
        return candidatos

    resultado = []
    por_consultar = []

    for nombre, cuenta in candidatos:
        # Whitelist: aprobado directo sin consultar Gemini
        if nombre in PLATOS_WHITELIST or any(
            nombre == wl or nombre.startswith(wl + ' ') or nombre.endswith(' ' + wl)
            for wl in PLATOS_WHITELIST
        ):
            resultado.append((nombre, cuenta))
        elif nombre in _cache_es_plato:
            if _cache_es_plato[nombre]:
                resultado.append((nombre, cuenta))
            # else: rechazado en cache, no incluir
        else:
            por_consultar.append((nombre, cuenta))

    if not por_consultar:
        return resultado

    # Sin Gemini: heurística estricta (en caso de duda, rechazar)
    if not _GEMINI_KEY:
        for nombre, cuenta in por_consultar:
            if PATRONES_NO_PLATO.search(nombre):
                _cache_es_plato[nombre] = False
            else:
                _cache_es_plato[nombre] = True
                resultado.append((nombre, cuenta))
        return resultado

    # Con Gemini: consulta en lote con prompt más estricto
    nombres_consultar = [n for n, _ in por_consultar]
    prompt = (
        "Eres un clasificador estricto. Analiza estos términos extraídos de reseñas "
        f"de un restaurante: {nombres_consultar}\n\n"
        "Clasifica cada uno en EXACTAMENTE una categoría:\n"
        "  es_plato    → nombre de un plato, comida, bebida o ingrediente concreto\n"
        "                Ejemplos: pulpo, tikka masala, paella, croquetas, vino tinto,\n"
        "                          risotto de hongos, tarta de queso, huevos rotos\n"
        "  no_es_plato → todo lo demás: adjetivos, frases de experiencia, nombres de\n"
        "                personas, servicios, lugares, o cualquier cosa que no sea\n"
        "                específicamente algo que se come o bebe\n"
        "                Ejemplos: lugar muy bonito, servicio de melvin, mejores brunch,\n"
        "                          melvin y vicente, camarero melvin, platos y cócteles,\n"
        "                          brunch de madrid, barco de sushi\n\n"
        "REGLA CLAVE: si contiene nombres propios de personas, lugares turísticos,\n"
        "adjetivos valorativos o descripciones de servicio → no_es_plato.\n"
        "En caso de duda → no_es_plato.\n\n"
        "Responde ÚNICAMENTE con JSON válido (sin texto extra, sin markdown):\n"
        '{"es_plato": [...], "no_es_plato": [...]}\n'
        "Todos los términos deben aparecer en una de las dos listas."
    )
    parsed = _gemini_json(prompt)
    aprobados  = set(parsed.get("es_plato", []))
    rechazados = set(parsed.get("no_es_plato", []))

    rechazados_log = []

    for nombre, cuenta in por_consultar:
        if nombre in aprobados:
            _cache_es_plato[nombre] = True
            resultado.append((nombre, cuenta))
        else:
            _cache_es_plato[nombre] = False
            rechazados_log.append(nombre)

    # Guardar para logging posterior (si 0 platos)
    if not hasattr(filtrar_platos_con_gemini, '_ultimo_rechazo'):
        filtrar_platos_con_gemini._ultimo_rechazo = []
    filtrar_platos_con_gemini._ultimo_rechazo = rechazados_log

    return resultado

# Adjetivos que NO pueden iniciar un bigrama de plato
ADJETIVOS_INICIO = {
    'buena','bueno','buenos','buenas','gran','grande','grandes',
    'muy','mejor','peor','rica','rico','ricos','ricas',
    'excelente','excelentes','increíble','increíbles','espectacular',
    'fantástico','fantástica','maravilloso','maravillosa','perfecto','perfecta',
    'impecable','excepcional','deliciosa','delicioso','sabroso','sabrosa',
    'calidad','precio','buen','súper','super',
    # Sustantivos que generan bigramas de servicio/lugar, no de plato
    'servicio','camarero','camarera','barco','lugar','ambiente','trato',
    'mejor','mejores','peores',
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
}

# ─────────────────────────────────────────────
# GEMINI — solo para nombres y normalización
# ─────────────────────────────────────────────
_GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_MODEL = "gemini-2.5-flash"
_cache_nombres = {}

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
            return texto_resp
        except Exception as e:
            if any(c in str(e) for c in ['503','429','500']) and intento < retries-1:
                espera = 2 ** (intento + 1)
                print(f"      [Gemini] reintentando en {espera}s...")
                _time.sleep(espera)
            else:
                return ""
    return ""

def _gemini_json(prompt):
    """Llama a Gemini y parsea JSON de la respuesta."""
    if not _GEMINI_KEY:
        return {}
    data = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
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

def clasificar_nombres(candidatos: list) -> set:
    """Devuelve solo los que son nombres de personas."""
    por_consultar = [t for t in candidatos if t not in _cache_nombres]
    if por_consultar:
        prompt = (
            f"Clasifica estas palabras: {por_consultar}\n"
            "nombres_persona = nombres propios de personas\n"
            "no_nombre = todo lo demas (adjetivos, sustantivos, lugares, comida)\n"
            "Si dudas, pon en no_nombre.\n"
            'Responde SOLO con: {"nombres_persona": [...], "no_nombre": [...]}'
        )
        parsed = _gemini_json(prompt)
        for t in parsed.get("nombres_persona", []):
            if t in por_consultar:
                _cache_nombres[t] = True
        for t in parsed.get("no_nombre", []):
            if t in por_consultar:
                _cache_nombres[t] = False
        for t in por_consultar:
            if t not in _cache_nombres:
                _cache_nombres[t] = None
    return {t for t in candidatos if _cache_nombres.get(t) is True}

def normalizar_platos(platos: list) -> list:
    """Corrige ortografía y fusiona variantes del mismo plato."""
    if not platos:
        return platos

    # Deduplicar ANTES de Gemini: misma clave -> sumar conteos
    # (ocurre cuando el mismo bigrama se genera con distinta ventana)
    vistos: dict = {}
    for nombre, cuenta in platos:
        vistos[nombre] = vistos.get(nombre, 0) + cuenta
    platos = sorted(vistos.items(), key=lambda x: -x[1])

    if not _GEMINI_KEY:
        return platos

    nombres = [t for t, _ in platos]
    prompt = (
        f"Normaliza estos nombres de platos: {nombres}\n"
        "- Corrige ortografia y acentos (nan->naan, bunuelos->buñuelos)\n"
        "- Fusiona variantes del MISMO plato con el nombre canónico\n"
        "  (ej: pollo tikka + tikka masala -> ambos pasan a ser tikka masala)\n"
        "- Minusculas, no traduzcas\n"
        "- Devuelve exactamente el mismo numero de elementos en el mismo orden\n"
        'Responde SOLO con: {"nombres": ["nombre1", "nombre2"]}'
    )
    parsed = _gemini_json(prompt)
    corregidos = parsed.get("nombres", [])
    if len(corregidos) != len(platos):
        return platos

    # Fusionar conteos de variantes que Gemini unificó al mismo nombre canónico
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
    # Garantizar que todos los elementos son strings no vacíos
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

def extraer_platos(serie_reviews, n=5, nombre_restaurante=""):
    """Tokens + bigramas con filtro NO_PLATOS. Sin Gemini."""
    freq_resenas = Counter()
    freq_total   = Counter()
    for r in serie_reviews:
        texto   = limpiar(str(r))
        palabras = re.findall(r"[a-záéíóúüñ]+", texto)
        # Tokens simples
        tokens_set = set()
        for t in palabras:
            if len(t) >= 3 and t not in STOP_ES and t not in NO_PLATOS:
                freq_total[t] += 1
                tokens_set.add(t)
        for t in tokens_set:
            freq_resenas[t] += 1
        # Bigramas (saltando stopwords, máx distancia 3)
        contenido = [(i,p) for i,p in enumerate(palabras)
                     if p not in STOP_ES and len(p) >= 3]
        vistos = set()
        for idx in range(len(contenido)-1):
            pos_a, a = contenido[idx]
            pos_b, b = contenido[idx+1]
            if pos_b - pos_a <= 3 and a not in ADJETIVOS_INICIO:
                bg = " ".join(palabras[pos_a:pos_b+1])
                if bg not in NO_PLATOS:
                    freq_total[bg] += 1
                    if bg not in vistos:
                        freq_resenas[bg] += 1
                        vistos.add(bg)

    # Umbral primario: >= 2 menciones (con 90 reseñas esto es lo normal)
    candidatos_2 = [(t, freq_resenas[t]) for t in freq_resenas if freq_resenas[t] >= 2]
    candidatos_2.sort(key=lambda x: -x[1])

    # Umbral fallback: >= 1 mención (para completar si no llegamos a n=5)
    candidatos_1 = [(t, freq_resenas[t]) for t in freq_resenas if freq_resenas[t] == 1]
    candidatos_1.sort(key=lambda x: -x[1])

    candidatos = candidatos_2 + candidatos_1  # fallback al final

    # Eliminar tokens simples cubiertos por bigramas
    bigramas = {t for t,_ in candidatos if ' ' in t}
    palabras_en_bigramas = set()
    for bg in bigramas:
        for w in re.findall(r"[a-záéíóúüñ]{3,}", bg):
            palabras_en_bigramas.add(w)

    # Construir lista con margen (n*3) antes del filtro Gemini
    platos_sin_filtrar = []
    for t, c in candidatos:
        if ' ' not in t and t in palabras_en_bigramas:
            continue
        # Rechazar heurísticamente antes de llamar a Gemini
        if PATRONES_NO_PLATO.search(t):
            continue
        platos_sin_filtrar.append((t, c))
        if len(platos_sin_filtrar) >= n * 3:  # margen: triple del necesario
            break

    # Filtro Gemini (o whitelist si no hay key) — pedir el doble de margen
    platos_filtrados = filtrar_platos_con_gemini(platos_sin_filtrar, nombre_restaurante)

    # Orden descendente GARANTIZADO por frecuencia (estable: empates mantienen orden de entrada)
    platos_filtrados.sort(key=lambda x: -x[1])

    # Recortar a n
    platos = platos_filtrados[:n]

    # Normalizar (puede fusionar duplicados, reordena internamente)
    platos_norm = normalizar_platos(platos)

    # Re-ordenar tras normalización para garantizar descendente
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
    """Detecta nombres de personas via Gemini."""
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

    # Dimensiones
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

    # Platos (sin Gemini para clasificar)
    platos_top  = extraer_platos(df_r['Review'], n=5, nombre_restaurante=nombre)
    platos_str  = ', '.join([f"{p}({c})" for p,c in platos_top])
    platos_set  = {p for p,_ in platos_top}

    # Personal (Gemini solo para clasificar nombres)
    personal    = extraer_personal(df_r['Review'], platos_set, n)
    personal_str= ', '.join([f"{t.capitalize()}({c})" for t,c in personal])

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
        'top5_platos':          platos_str,
        'personal_destacado':   personal_str,
        'terminos_tfidf':       tfidf_str,
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
    print(f"✔ Gemini configurado: {_GEMINI_MODEL} (nombres + normalización)")
else:
    print("⚠️  Sin GEMINI_API_KEY — nombres y normalización desactivados")
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

ids_todos = sorted(df['Id_Restaurante'].unique())

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
ids_procesados    = set()
ids_incompletos   = set()  # procesados pero con < 5 platos → reprocesar
filas_resumen     = []
filas_resenas     = []

if os.path.exists(OUTPUT_CSV):
    df_prev = pd.read_csv(OUTPUT_CSV)
    # Contar platos: la columna tiene formato "plato1(n), plato2(n), ..."
    # Un restaurante con k platos tiene k comas+1 elementos (o 0 si está vacío)
    def _contar_platos(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        return len(str(val).split(','))
    df_prev['_n_platos'] = df_prev['top5_platos'].apply(_contar_platos)
    ids_completos   = set(df_prev[df_prev['_n_platos'] >= 5]['id_restaurante'].astype(int))
    ids_incompletos = set(df_prev[df_prev['_n_platos'] <  5]['id_restaurante'].astype(int))
    ids_procesados  = ids_completos  # solo saltamos los que ya tienen 5

    # Cargar filas completas en memoria; las incompletas se sobreescribirán
    filas_resumen = df_prev[df_prev['id_restaurante'].astype(int).isin(ids_completos)] \
                        .drop(columns=['_n_platos']).to_dict('records')
    print(f"CSV existente: {len(ids_completos)} completos (5 platos), "
          f"{len(ids_incompletos)} incompletos → se reprocesarán.")
else:
    print("No existe CSV previo — procesando todo desde cero.")

if os.path.exists(OUTPUT_RESENAS):
    df_res_prev = pd.read_csv(OUTPUT_RESENAS)
    if ids_procesados:
        # Conservar solo las reseñas de restaurantes completos
        filas_resenas = [df_res_prev[df_res_prev['Id_Restaurante'].astype(int).isin(ids_procesados)]]

ids_pendientes = [i for i in ids_todos if int(i) not in ids_procesados]
print(f"Total: {len(ids_todos)} | Completos: {len(ids_procesados)} | "
      f"Pendientes: {len(ids_pendientes)} "
      f"({len(ids_incompletos)} reprocesados + {len(ids_pendientes)-len(ids_incompletos)} nuevos)\n")

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
        _guardar_cache_platos()  # persistir cache Gemini tras cada restaurante

        print(f"    ✔ {fila['pct_positivo']}% pos | ★{fila['avg_estrellas_modelo']} | {fila['top5_platos'][:70]}")

    except Exception as e:
        import traceback
        print(f"    ✗ Error: {e}")
        traceback.print_exc()

print(f"\n{'='*60}\n¡Completado!\n  → {OUTPUT_CSV}\n  → {OUTPUT_RESENAS}")
