"""
generar_geo.py
──────────────
Lee ranking.csv, geocodifica las direcciones de los restaurantes
con Nominatim (OpenStreetMap, gratis, sin API key) y genera
restaurantes_geo.csv con columnas: Id_Restaurante, latitud, longitud.

Uso:
    python generar_geo.py

Requiere:
    pip install geopy

El script es incremental: si restaurantes_geo.csv ya existe,
solo geocodifica los IDs que faltan y añade las filas nuevas.
"""

import os
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ── Configuración ─────────────────────────────────────────────────────────────
RANKING_CSV   = os.path.join(os.path.dirname(__file__), "ranking.csv")
GEO_CSV       = os.path.join(os.path.dirname(__file__), "restaurantes_geo.csv")
PAUSA_SEGUNDOS = 1.1   # Nominatim exige máximo 1 req/seg
MAX_INTENTOS   = 3
ID_MAX         = None  # None = todos; pon un número para limitar (ej. 165)
# ─────────────────────────────────────────────────────────────────────────────

geolocator = Nominatim(user_agent="RestaurantesMadridTFM/1.0")


def geocodificar(direccion: str, nombre: str) -> tuple:
    """Intenta geocodificar una dirección en Madrid. Devuelve (lat, lon) o (None, None)."""
    candidatos = [
        f"{direccion}, Madrid, España",
        f"{nombre}, Madrid, España",
        f"{direccion}, Madrid",
    ]
    for consulta in candidatos:
        for intento in range(MAX_INTENTOS):
            try:
                loc = geolocator.geocode(consulta, timeout=5)
                if loc:
                    return round(loc.latitude, 7), round(loc.longitude, 7)
                break  # no encontrado, probar siguiente candidato
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"    ⚠️  Intento {intento+1}/{MAX_INTENTOS}: {e}")
                time.sleep(2)
        time.sleep(PAUSA_SEGUNDOS)
    return None, None


def main():
    # Cargar ranking
    ranking = pd.read_csv(RANKING_CSV, sep=";", skiprows=1)
    ranking.columns = ranking.columns.str.strip()

    if ID_MAX:
        ranking = ranking[ranking["Id_Restaurante"] <= ID_MAX]

    print(f"Restaurantes en ranking: {len(ranking)}")

    # Cargar geo existente (incremental)
    if os.path.exists(GEO_CSV):
        geo_prev = pd.read_csv(GEO_CSV)
        ids_ya = set(geo_prev["Id_Restaurante"].tolist())
        filas = geo_prev.to_dict("records")
        print(f"Ya geocodificados: {len(ids_ya)}")
    else:
        ids_ya = set()
        filas = []

    pendientes = ranking[~ranking["Id_Restaurante"].isin(ids_ya)]
    print(f"Pendientes: {len(pendientes)}\n")

    for _, row in pendientes.iterrows():
        id_r     = int(row["Id_Restaurante"])
        nombre   = str(row.get("Restaurante", "") or "")
        direccion = str(row.get("Dirección", "") or "")

        if not direccion or direccion == "nan":
            print(f"  [{id_r}] {nombre} — sin dirección, saltando")
            filas.append({"Id_Restaurante": id_r, "latitud": None, "longitud": None})
            continue

        print(f"  [{id_r}] {nombre} ({direccion}) ... ", end="", flush=True)
        lat, lon = geocodificar(direccion, nombre)

        if lat:
            print(f"✓ {lat}, {lon}")
        else:
            print("✗ no encontrado")

        filas.append({"Id_Restaurante": id_r, "latitud": lat, "longitud": lon})

        # Guardar progreso tras cada restaurante (por si se interrumpe)
        pd.DataFrame(filas).sort_values("Id_Restaurante").to_csv(GEO_CSV, index=False)
        time.sleep(PAUSA_SEGUNDOS)

    # Guardado final ordenado
    resultado = pd.DataFrame(filas).sort_values("Id_Restaurante")
    resultado.to_csv(GEO_CSV, index=False)

    con_coords = resultado["latitud"].notna().sum()
    print(f"\n✅ Completado: {con_coords}/{len(resultado)} restaurantes geocodificados")
    print(f"   → {GEO_CSV}")


if __name__ == "__main__":
    main()
