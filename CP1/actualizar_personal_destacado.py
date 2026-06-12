"""
actualizar_personal_destacado.py
Actualiza SOLO la columna personal_destacado en analisis_restaurantes.csv
usando patrones de mención directa en resenas_unificadas.csv.
Sin Gemini, sin reprocesar BERT. Ejecutar desde la carpeta del proyecto.
"""
import pandas as pd, re, unicodedata, os
from collections import Counter

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_RESENAS    = os.path.join(BASE_DIR, "resenas_unificadas.csv")
CSV_ANALISIS   = os.path.join(BASE_DIR, "analisis_restaurantes.csv")

print(f"Cargando reseñas: {CSV_RESENAS}")
df_resenas  = pd.read_csv(CSV_RESENAS)
print(f"Cargando análisis: {CSV_ANALISIS}")
df_analisis = pd.read_csv(CSV_ANALISIS)
print(f"  {len(df_resenas)} reseñas · {len(df_analisis)} restaurantes\n")

# ── Palabras que NUNCA son nombres de persona ────────────────────────────────
STOP_PERSONAL = {
    'el','la','los','las','un','una','de','del','al','en','con','por','que',
    'nos','les','fue','es','era','son','muy','todo','bien','mal','hay','han',
    'comida','servicio','lugar','sitio','restaurante','mesa','plato','trato',
    'precio','ambiente','terraza','carta','vino','agua','pan','cafe','postre',
    'excelente','bueno','buena','malo','mala','genial','increible','perfecto',
    'impecable','fantastico','amable','atento','atenta','profesional','maravilloso',
    'siempre','nunca','tambien','ademas','mucho','poco','nada','algo',
    'personal','equipo','staff','gente','chico','chica','senor','senora',
    'madrid','calle','verdad','parte','lado','vez','veces','dia','noche',
    'tarde','atencion','experiencia','visita','reserva','encanto','rapido',
    'recomendable','recomendado','volveremos','volvere','volveria','repetir',
    'seguro','duda','pronto','gracias','especialmente','destacar','estupendo',
    'espectacular','inmejorable','fenomenal','extraordinario','sobresaliente',
    'excepcional','agradable','simpatico','simpatica','amables','atentos',
    'educados','correcto','todos','algunas','algunos','este','esta','estos',
    'aqui','alli','cuando','como','para','mas','menos','tan','tanto','igual',
    'camarero','camarera','chef','cocinero','maitre','gerente','encargado',
    'dueno','duena','propietario','socio','sala','cocina','local','negocio',
    'hora','minuto','momento','primera','segunda','ultima','proxima',
    'sabor','textura','calidad','cantidad','presentacion',
    'sushi','pizza','vinos','cerveza','mariscos','tapas','pasta','arabe',
    'asiatica','cilantro','atendio','recibido','encantador','encantadora',
    'vermut','amor','guerrero','ella','ellos','ellas','nosotros','vosotros',
    'tod','mare','pero','sus','sin','hoy','sale','tampoco','nuestro','resulto',
    'extraordinar','platos','volvemos','super','rapidos','atentas',
}

# ── Patrones de mención directa en contexto de servicio ─────────────────────
PATRONES = [
    r'gracias\s+a\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})(?:\s+y\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12}))?',
    r'de\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})\s+(?:fue|estuvo)\s+(?:excelente|impecable|genial|increible|fantastico|perfecto|excepcional|maravilloso|sobresaliente|extraordinario)',
    r'([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})\s+(?:fue|es|estuvo)\s+(?:muy\s+|super\s+)?(?:amable|atento|atenta|profesional|excepcional|genial|increible|fantastico|impecable|simpatico|simpatica|encantador|encantadora)',
    r'(?:el\s+camarero|la\s+camarera|el\s+maitre|el\s+chef|el\s+sommelier|el\s+encargado|la\s+encargada|el\s+gerente)\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})',
    r'(?:recomendaciones?|consejos?|sugerencias?)\s+de\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})',
    r'nos\s+(?:atendio|ayudo|explico|recomendo|asesoro|recibio)\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})',
    r'(?:destacar|mencionar|agradecer|felicitar)\s+a\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})',
    r'atendidos?\s+por\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})',
    r'(?:labor|trabajo|profesionalidad|dedicacion|amabilidad)\s+de\s+([a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{3,12})',
]

def norm(s):
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')

def extraer_personal(resenas, n=3):
    MIN_RESENAS = 2
    contador = Counter()
    for resena in resenas:
        texto = norm(str(resena or ''))
        encontrados = set()
        for patron in PATRONES:
            for match in re.finditer(patron, texto):
                for grupo in match.groups():
                    if grupo:
                        nombre = grupo.strip()
                        if (nombre not in STOP_PERSONAL
                                and len(nombre) >= 3
                                and len(nombre) <= 12
                                and nombre.isalpha()):
                            encontrados.add(nombre)
        for nombre in encontrados:
            contador[nombre] += 1
    return [f"{nombre.capitalize()}({count})"
            for nombre, count in contador.most_common(n * 2)
            if count >= MIN_RESENAS][:n]

# ── Procesar cada restaurante ────────────────────────────────────────────────
resultados = {}
ids = df_resenas['Id_Restaurante'].unique()
print(f"Procesando {len(ids)} restaurantes...")

for rid in sorted(ids):
    resenas = df_resenas[df_resenas['Id_Restaurante'] == rid]['Review'].tolist()
    nombres = extraer_personal(resenas)
    resultados[int(rid)] = ', '.join(nombres)
    if nombres:
        nombre_rest = df_resenas[df_resenas['Id_Restaurante'] == rid]['Restaurante'].iloc[0]
        print(f"  [{rid:>3}] {nombre_rest}: {', '.join(nombres)}")

# ── Actualizar CSV ───────────────────────────────────────────────────────────
df_analisis['personal_destacado'] = df_analisis['id_restaurante'].apply(
    lambda x: resultados.get(int(x), '')
)

con_personal = (df_analisis['personal_destacado'] != '').sum()
df_analisis.to_csv(CSV_ANALISIS, index=False)
print(f"\n✔ {con_personal}/165 restaurantes con personal identificado")
print(f"✔ CSV actualizado: {CSV_ANALISIS}")
