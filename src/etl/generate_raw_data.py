"""
SentinelaIA Nariño — Generador de datos sintéticos para data/raw/
Ejecutar desde la raíz del proyecto:
    python src/etl/generate_raw_data.py

Genera archivos .xlsx con formato idéntico al que esperan
clean_dane.py y clean_salud.py.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]
DANE_DIR = BASE_DIR / "data" / "raw" / "dane"
SALUD_DIR = BASE_DIR / "data" / "raw" / "salud"

# Municipios foco: código DIVIPOLA completo (5 dígitos) y población base 2023
MUNICIPIOS = [
    {"dp": 52, "mpio_3": 1,   "divipola": 52001, "nombre": "PASTO",       "pob": 450000},
    {"dp": 52, "mpio_3": 207, "divipola": 52207, "nombre": "CONSACA",     "pob": 11000},
    {"dp": 52, "mpio_3": 381, "divipola": 52381, "nombre": "LA FLORIDA",  "pob": 12000},
    {"dp": 52, "mpio_3": 480, "divipola": 52480, "nombre": "NARINO",      "pob": 28000},
    {"dp": 52, "mpio_3": 683, "divipola": 52683, "nombre": "SANDONA",     "pob": 22000},
    {"dp": 52, "mpio_3": 885, "divipola": 52885, "nombre": "YACUANQUER",  "pob": 10000},
]

GROWTH = 0.015


# ── 1. DANE ────────────────────────────────────────────────────
# clean_dane.py lee:
#   pd.read_excel(path, sheet_name="PobMunicipalxÁrea", header=7)
#   Columnas requeridas (post-normalize): dp, dpnom, mpio, dpmp, ano,
#     area_geografica, total
#   mpio → pd.to_numeric → zfill(5) → cod_divipola  (debe ser DIVIPOLA 5 dígitos)
#   dpmp → clean_text → municipio (luego se sobreescribe con map)
#   ano  → anio
#   area_geografica debe contener "TOTAL" para pasar el filtro
#   Filtra cod_divipola ∈ TARGET_CODES y anio ∈ [2023,2024,2025]

def generate_dane():
    DANE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in MUNICIPIOS:
        for i, year in enumerate([2023, 2024, 2025]):
            pob = int(m["pob"] * (1 + GROWTH) ** i)
            rows.append({
                "DP": m["dp"],
                "DPNOM": "NARIÑO",
                "MPIO": m["divipola"],       # ← DIVIPOLA completo (5 dígitos)
                "DPMP": m["nombre"],
                "AÑO": year,
                "Área Geográfica": "Total",  # clean_dane filtra por "TOTAL"
                "Total": pob,
            })
            # Fila cabecera (descartada por el filtro, da realismo)
            rows.append({
                "DP": m["dp"],
                "DPNOM": "NARIÑO",
                "MPIO": m["divipola"],
                "DPMP": m["nombre"],
                "AÑO": year,
                "Área Geográfica": "Cabecera Municipal",
                "Total": int(pob * 0.65),
            })

    df = pd.DataFrame(rows)
    out = DANE_DIR / "poblacion_municipal.xlsx"

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        # startrow=7 → header en fila 7 (0-indexed), datos desde fila 8
        # Esto coincide con header=7 en pd.read_excel
        df.to_excel(w, sheet_name="PobMunicipalxÁrea", index=False, startrow=7)

    print(f"  OK {out.relative_to(BASE_DIR)}  ({len(df)} filas)")


# ── Helpers estacionales ───────────────────────────────────────

def _peso_eda(sem):
    return 2.0 if (10 <= sem <= 20 or 35 <= sem <= 45) else 1.0

def _peso_ira(sem):
    return 2.5 if 15 <= sem <= 30 else 1.0

def _fecha_en_semana(year, week):
    lunes = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
    return lunes + timedelta(days=int(np.random.randint(0, 7)))


# ── 2. Salud EDA (998) ────────────────────────────────────────
# clean_salud.py → process_eda_file busca (post-normalize):
#   cod_mun / cod_mun_  → DIVIPOLA completo, zfill(5)
#   fec_not             → fecha
#   semana              → semana epidemiológica
#   ano / a_o / anio    → año
#   nom_eve             → nombre evento
#   nmun_proce / nmun_notif → municipio
#   ndep_proce / ndep_notif → departamento
#   cas_conc (opcional), hombres, mujeres → casos

def generate_eda(year, n=600):
    SALUD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    per_mun = n // len(MUNICIPIOS)

    for m in MUNICIPIOS:
        semanas = np.random.choice(range(1, 53), size=per_mun, replace=True)
        for sem in semanas:
            w = _peso_eda(int(sem))
            fecha = _fecha_en_semana(year, int(sem))
            rows.append({
                "COD_MUN_": m["divipola"],
                "FEC_NOT": fecha.strftime("%Y-%m-%d"),
                "SEMANA": int(sem),
                "ANO": year,
                "NOM_EVE": "MORBILIDAD POR EDA",
                "NMUN_PROCE": m["nombre"],
                "NDEP_PROCE": "NARIÑO",
                "HOMBRES": max(0, int(np.random.poisson(3 * w))),
                "MUJERES": max(0, int(np.random.poisson(3 * w))),
            })

    df = pd.DataFrame(rows[:n])
    out = SALUD_DIR / f"Datos_{year}_998.xlsx"
    df.to_excel(out, index=False, sheet_name="Datos", engine="openpyxl")
    print(f"  OK {out.relative_to(BASE_DIR)}  ({len(df)} filas)")


# ── 3. Salud IRA (346) ────────────────────────────────────────
# clean_salud.py → process_346_file busca (post-normalize):
#   cod_dpto_n              → depto (2 dígitos)
#   cod_mun_n               → municipal (3 dígitos)
#   build_divipola combina: zfill(depto,2) + zfill(mun,3) = DIVIPOLA
#   fec_not / fec_con / ini_sin → fecha
#   semana                  → semana
#   ano / anio              → año
#   municipio_notificacion  → municipio
#   departamento_notificacion → departamento
#   nombre_evento           → evento
#   confirmados             → casos

def generate_ira(year=2024, n=600):
    SALUD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    per_mun = n // len(MUNICIPIOS)

    for m in MUNICIPIOS:
        semanas = np.random.choice(range(1, 53), size=per_mun, replace=True)
        for sem in semanas:
            w = _peso_ira(int(sem))
            fecha = _fecha_en_semana(year, int(sem))
            rows.append({
                "COD_DPTO_N": m["dp"],        # 52 → zfill(2) → "52"
                "COD_MUN_N": m["mpio_3"],     # 1 → zfill(3) → "001"
                "FEC_NOT": fecha.strftime("%Y-%m-%d"),
                "SEMANA": int(sem),
                "ANO": year,
                "MUNICIPIO_NOTIFICACION": m["nombre"],
                "DEPARTAMENTO_NOTIFICACION": "NARIÑO",
                "NOMBRE_EVENTO": "INFECCION RESPIRATORIA AGUDA IRA",
                "CONFIRMADOS": max(1, int(np.random.poisson(4 * w))),
            })

    df = pd.DataFrame(rows[:n])
    out = SALUD_DIR / f"Datos_{year}_346.xlsx"
    df.to_excel(out, index=False, sheet_name="Datos", engine="openpyxl")
    print(f"  OK {out.relative_to(BASE_DIR)}  ({len(df)} filas)")


# ── MAIN ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SentinelaIA — Generación de datos sintéticos")
    print("=" * 60)

    print("\n[1] DANE poblacion_municipal.xlsx ...")
    generate_dane()

    print("\n[2] Salud EDA 2023 ...")
    generate_eda(2023, 600)

    print("\n[3] Salud EDA 2024 ...")
    generate_eda(2024, 600)

    print("\n[4] Salud IRA 2024 ...")
    generate_ira(2024, 600)

    print("\n" + "=" * 60)
    print("  Archivos generados correctamente")
    print("=" * 60)


if __name__ == "__main__":
    main()
