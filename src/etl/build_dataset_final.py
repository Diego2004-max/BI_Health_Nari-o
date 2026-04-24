from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FINAL_DIR = BASE_DIR / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

SALUD_FILE = PROCESSED_DIR / "salud_municipio_semana.csv"
DANE_FILE = PROCESSED_DIR / "dane_poblacion_2023_2025_galeras.csv"

TARGET_CODES = {
    "52001": "PASTO",
    "52207": "CONSACA",
    "52381": "LA FLORIDA",
    "52480": "NARINO",
    "52683": "SANDONA",
    "52885": "YACUANQUER",
}

def main():
    if not SALUD_FILE.exists():
        raise FileNotFoundError(f"No existe: {SALUD_FILE}")
    if not DANE_FILE.exists():
        raise FileNotFoundError(f"No existe: {DANE_FILE}")

    salud = pd.read_csv(SALUD_FILE)
    dane = pd.read_csv(DANE_FILE)

    salud["cod_divipola"] = salud["cod_divipola"].astype(str).str.zfill(5)
    salud["anio"] = pd.to_numeric(salud["anio"], errors="coerce").astype("Int64")
    salud["semana_epidemiologica"] = pd.to_numeric(
        salud["semana_epidemiologica"], errors="coerce"
    ).astype("Int64")
    salud["casos"] = pd.to_numeric(salud["casos"], errors="coerce").fillna(0)

    dane["cod_divipola"] = dane["cod_divipola"].astype(str).str.zfill(5)
    dane["anio"] = pd.to_numeric(dane["anio"], errors="coerce").astype("Int64")
    dane["poblacion"] = pd.to_numeric(dane["poblacion"], errors="coerce")

    # Nos quedamos con EDA, que es la base consistente actual
    salud = salud[salud["evento_estandar"] == "EDA"].copy()

    years = sorted(salud["anio"].dropna().unique().tolist())
    weeks = list(range(1, 53))

    grid = pd.MultiIndex.from_product(
        [list(TARGET_CODES.keys()), years, weeks],
        names=["cod_divipola", "anio", "semana_epidemiologica"]
    ).to_frame(index=False)

    grid["municipio"] = grid["cod_divipola"].map(TARGET_CODES)
    grid["departamento"] = "NARINO"
    grid["evento_estandar"] = "EDA"
    grid["evento_origen"] = "MORBILIDAD POR EDA"

    salud_merge = salud[
        [
            "evento_estandar",
            "evento_origen",
            "cod_divipola",
            "municipio",
            "departamento",
            "anio",
            "semana_epidemiologica",
            "casos",
        ]
    ].copy()

    dataset = grid.merge(
        salud_merge,
        on=[
            "evento_estandar",
            "evento_origen",
            "cod_divipola",
            "municipio",
            "departamento",
            "anio",
            "semana_epidemiologica",
        ],
        how="left",
    )

    dataset["casos"] = dataset["casos"].fillna(0)

    dataset = dataset.merge(
        dane[["cod_divipola", "anio", "poblacion"]],
        on=["cod_divipola", "anio"],
        how="left",
    )

    dataset["tasa_x_100k"] = (dataset["casos"] / dataset["poblacion"]) * 100000
    dataset["tasa_x_100k"] = dataset["tasa_x_100k"].round(4)

    dataset = dataset.sort_values(
        ["cod_divipola", "anio", "semana_epidemiologica"]
    ).reset_index(drop=True)

    dataset.to_csv(
        FINAL_DIR / "dataset_final_eda_municipio_semana.csv",
        index=False,
        encoding="utf-8-sig",
    )

    resumen = (
        dataset.groupby(["municipio", "anio"], as_index=False)
        .agg(
            casos_totales=("casos", "sum"),
            poblacion=("poblacion", "max"),
            tasa_anual_x_100k=("tasa_x_100k", "sum"),
        )
        .sort_values(["municipio", "anio"])
    )
    resumen.to_csv(
        FINAL_DIR / "resumen_final_eda_anual.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Archivos generados:")
    print("-", FINAL_DIR / "dataset_final_eda_municipio_semana.csv")
    print("-", FINAL_DIR / "resumen_final_eda_anual.csv")

if __name__ == "__main__":
    main()