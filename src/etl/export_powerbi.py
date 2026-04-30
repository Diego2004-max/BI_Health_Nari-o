"""
SentinelaIA Nariño — Exportar CSV consolidado para Power BI
Ejecutar desde la raíz del proyecto:
    python src/etl/export_powerbi.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import warnings

# ── Rutas ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data" / "final" / "dataset_final_eda_municipio_semana.csv"
PRED_PATH = BASE_DIR / "data" / "final" / "predicciones_riesgo.csv"
OUTPUT_PATH = BASE_DIR / "data" / "final" / "sentinela_powerbi.csv"

# ── Mapeo de nivel de riesgo a color de semáforo ───────────────
SEMAFORO_MAP = {
    "bajo":  "Verde",
    "medio": "Amarillo",
    "alto":  "Rojo",
}


def fecha_lunes_iso(row):
    """Devuelve el lunes de la semana ISO a partir de anio + semana_epidemiologica."""
    try:
        # Formato ISO: %G = año ISO, %V = semana ISO, %u = día (1=lunes)
        return pd.to_datetime(
            f"{int(row['anio'])}-W{int(row['semana_epidemiologica']):02d}-1",
            format="%G-W%V-%u",
        )
    except Exception:
        return pd.NaT


def main():
    # ── 1. Leer dataset base ──────────────────────────────────
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset base: {DATASET_PATH}\n"
            "Ejecuta primero: python src/etl/build_dataset_final.py"
        )
    base = pd.read_csv(DATASET_PATH)
    print(f"[1] Dataset base cargado: {base.shape[0]:,} filas × {base.shape[1]} columnas")

    # ── 2. Leer predicciones ──────────────────────────────────
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de predicciones: {PRED_PATH}\n"
            "Ejecuta primero: python src/models/modelo_xgboost.py"
        )
    pred = pd.read_csv(PRED_PATH)
    print(f"[2] Predicciones cargadas: {pred.shape[0]:,} filas × {pred.shape[1]} columnas")

    # ── 3. LEFT JOIN ──────────────────────────────────────────
    join_keys = ["municipio", "semana_epidemiologica", "anio"]

    # Verificar municipios sin coincidencia
    municipios_base = set(base["municipio"].unique())
    municipios_pred = set(pred["municipio"].unique())
    solo_en_pred = municipios_pred - municipios_base
    if solo_en_pred:
        warnings.warn(
            f"⚠️  Municipios en predicciones que NO están en el dataset base "
            f"(se conservarán con valores NaN): {solo_en_pred}"
        )

    df = base.merge(
        pred[join_keys + ["nivel_riesgo_predicho", "probabilidad"]],
        on=join_keys,
        how="left",
    )
    print(f"[3] JOIN completado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # ── 4. Columnas calculadas para Power BI ──────────────────
    # 4a. fecha_semana: lunes de la semana ISO
    df["fecha_semana"] = df.apply(fecha_lunes_iso, axis=1)

    # 4b. semaforo_color
    df["semaforo_color"] = (
        df["nivel_riesgo_predicho"]
        .str.lower()
        .map(SEMAFORO_MAP)
        .fillna("Sin dato")
    )

    # 4c. probabilidad_pct (0–100, 1 decimal)
    df["probabilidad_pct"] = (df["probabilidad"] * 100).round(1)

    # 4d. municipio_label: legible para Power BI
    df["municipio_label"] = (
        df["municipio"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    print(f"[4] Columnas calculadas añadidas")

    # ── 5. Guardar CSV para Power BI ──────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[5] Archivo guardado: {OUTPUT_PATH}")

    # ── 6. Resumen de verificación ────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN — sentinela_powerbi.csv")
    print("=" * 60)
    print(f"  Forma:    {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  Columnas: {list(df.columns)}")
    print(f"\n  Primeras 3 filas:")
    print(df.head(3).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
