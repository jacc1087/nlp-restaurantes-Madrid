"""
añadir_direcciones.py
─────────────────────
Añade la columna 'direccion' al CSV de análisis de restaurantes
leyéndola del ranking.csv, sin ninguna llamada a APIs externas.

Uso:
    python3 añadir_direcciones.py
"""

import pandas as pd
import os

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RANKING_CSV = os.path.join(BASE_DIR, "ranking.csv")
ANALISIS_CSV = os.path.join(BASE_DIR, "analisis_restaurantes.csv")
OUTPUT_CSV  = os.path.join(BASE_DIR, "analisis_restaurantes.csv")  # sobreescribe

# ── Carga ─────────────────────────────────────────────────────────────────────
print("Cargando archivos...")

ranking  = pd.read_csv(RANKING_CSV, sep=";", skiprows=1)
analisis = pd.read_csv(ANALISIS_CSV)

print(f"  Ranking:  {len(ranking)} restaurantes, columnas: {ranking.columns.tolist()}")
print(f"  Análisis: {len(analisis)} restaurantes, columnas: {analisis.columns.tolist()}")

# ── Preparar columna de dirección ─────────────────────────────────────────────
direcciones = ranking[["Id_Restaurante", "Dirección"]].copy()
direcciones.columns = ["id_restaurante", "direccion"]
direcciones["id_restaurante"] = direcciones["id_restaurante"].astype(int)
direcciones["direccion"] = direcciones["direccion"].str.strip()

# ── Merge ─────────────────────────────────────────────────────────────────────
analisis["id_restaurante"] = analisis["id_restaurante"].astype(int)

# Insertar columna 'direccion' justo después de 'nombre_restaurante'
if "direccion" in analisis.columns:
    analisis = analisis.drop(columns=["direccion"])

analisis = analisis.merge(direcciones, on="id_restaurante", how="left")

# Reordenar: poner direccion como 3ª columna
cols = analisis.columns.tolist()
cols.remove("direccion")
pos = cols.index("nombre_restaurante") + 1 if "nombre_restaurante" in cols else 2
cols.insert(pos, "direccion")
analisis = analisis[cols]

# ── Resultado ─────────────────────────────────────────────────────────────────
sin_dir = analisis["direccion"].isna().sum()
if sin_dir:
    print(f"\n⚠️  {sin_dir} restaurantes sin dirección encontrada:")
    print(analisis[analisis["direccion"].isna()][["id_restaurante","nombre_restaurante"]])

analisis.to_csv(OUTPUT_CSV, index=False)
print(f"\n✔ Guardado en: {OUTPUT_CSV}")
print(f"  {len(analisis)} restaurantes con dirección añadida.")
print(f"\nEjemplo:")
print(analisis[["id_restaurante","nombre_restaurante","direccion"]].head(5).to_string(index=False))
