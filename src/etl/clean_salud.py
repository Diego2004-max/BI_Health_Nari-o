from pathlib import Path
import pandas as pd
import unicodedata
import re

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "salud"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Municipios foco Galeras
TARGET_CODES = {
    "52001": "PASTO",
    "52207": "CONSACA",
    "52381": "LA FLORIDA",
    "52480": "NARINO",
    "52683": "SANDONA",
    "52885": "YACUANQUER",
}

FILES = [
    RAW_DIR / "Datos_2023_998.xlsx",
    RAW_DIR / "Datos_2024_998.xlsx",
    RAW_DIR / "Datos_2024_346.xlsx",
]

def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("utf-8")
    value = re.sub(r"\s+", " ", value)
    return value.upper()

def normalize_col(col):
    col = str(col).strip()
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def zfill_series(series, n):
    return series.astype("Int64").astype(str).str.replace("<NA>", "", regex=False).str.zfill(n)

def build_divipola(depto_series, mun_series):
    depto = safe_to_numeric(depto_series)
    mun = safe_to_numeric(mun_series)
    return zfill_series(depto, 2) + zfill_series(mun, 3)

def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def process_eda_file(path: Path):
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [normalize_col(c) for c in df.columns]

    col_cod = find_first_existing(df, ["cod_mun", "cod_mun_"])
    col_fecha = find_first_existing(df, ["fec_not"])
    col_semana = find_first_existing(df, ["semana"])
    col_anio = find_first_existing(df, ["ano", "a_o", "anio"])
    col_casos = find_first_existing(df, ["cas_conc"])
    col_hombres = find_first_existing(df, ["hombres"])
    col_mujeres = find_first_existing(df, ["mujeres"])
    col_mun = find_first_existing(df, ["nmun_proce", "nmun_notif"])
    col_dep = find_first_existing(df, ["ndep_proce", "ndep_notif"])
    col_evento = find_first_existing(df, ["nom_eve"])

    out = pd.DataFrame(index=df.index)
    out["fuente_archivo"] = path.name
    out["evento_estandar"] = "EDA"
    if col_evento:
        out["evento_origen"] = df[col_evento].astype(str)
    else:
        out["evento_origen"] = "MORBILIDAD POR EDA"

    out["cod_divipola"] = (
        safe_to_numeric(df[col_cod]).astype("Int64").astype(str).str.replace("<NA>", "", regex=False).str.zfill(5)
    )
    out["fecha_evento"] = safe_to_datetime(df[col_fecha]) if col_fecha else pd.NaT
    out["anio"] = safe_to_numeric(df[col_anio]) if col_anio else out["fecha_evento"].dt.year
    out["semana_epidemiologica"] = safe_to_numeric(df[col_semana]) if col_semana else out["fecha_evento"].dt.isocalendar().week.astype("Int64")
    out["departamento"] = df[col_dep].apply(clean_text) if col_dep else None
    out["municipio"] = df[col_mun].apply(clean_text) if col_mun else None

    if col_casos:
        out["casos"] = safe_to_numeric(df[col_casos])
    else:
        hombres = safe_to_numeric(df[col_hombres]) if col_hombres else 0
        mujeres = safe_to_numeric(df[col_mujeres]) if col_mujeres else 0
        out["casos"] = hombres.fillna(0) + mujeres.fillna(0)

    if col_hombres or col_mujeres:
        hombres = safe_to_numeric(df[col_hombres]) if col_hombres else 0
        mujeres = safe_to_numeric(df[col_mujeres]) if col_mujeres else 0
        total_hm = hombres.fillna(0) + mujeres.fillna(0)
        out["casos"] = out["casos"].fillna(total_hm)

    return out

def process_346_file(path: Path):
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [normalize_col(c) for c in df.columns]

    col_mun_code = find_first_existing(df, ["cod_mun_n"])
    col_dep_code = find_first_existing(df, ["cod_dpto_n"])
    col_fecha = find_first_existing(df, ["fec_not", "fec_con", "ini_sin"])
    col_semana = find_first_existing(df, ["semana"])
    col_anio = find_first_existing(df, ["ano", "anio"])
    col_mun = find_first_existing(df, ["municipio_notificacion"])
    col_dep = find_first_existing(df, ["departamento_notificacion"])
    col_evento = find_first_existing(df, ["nombre_evento"])
    col_confirmados = find_first_existing(df, ["confirmados"])

    out = pd.DataFrame(index=df.index)
    out["fuente_archivo"] = path.name
    out["evento_estandar"] = "IRA_VIRUS_NUEVO_346"
    out["evento_origen"] = df[col_evento].astype(str) if col_evento else "IRA POR VIRUS NUEVO"
    out["cod_divipola"] = build_divipola(df[col_dep_code], df[col_mun_code])
    out["fecha_evento"] = safe_to_datetime(df[col_fecha]) if col_fecha else pd.NaT
    out["anio"] = safe_to_numeric(df[col_anio]) if col_anio else out["fecha_evento"].dt.year
    out["semana_epidemiologica"] = safe_to_numeric(df[col_semana]) if col_semana else out["fecha_evento"].dt.isocalendar().week.astype("Int64")
    out["departamento"] = df[col_dep].apply(clean_text) if col_dep else None
    out["municipio"] = df[col_mun].apply(clean_text) if col_mun else None
    out["casos"] = safe_to_numeric(df[col_confirmados]).fillna(1) if col_confirmados else 1

    return out

def filter_target_codes(df):
    return df[df["cod_divipola"].isin(TARGET_CODES.keys())].copy()

def fill_target_names(df):
    df["municipio"] = df["cod_divipola"].map(TARGET_CODES)
    df["departamento"] = "NARINO"
    return df

def aggregate_weekly(df):
    keys = ["evento_estandar", "evento_origen", "cod_divipola", "municipio", "departamento", "anio", "semana_epidemiologica"]
    return (
        df.groupby(keys, dropna=False, as_index=False)["casos"]
        .sum()
        .sort_values(["evento_estandar", "cod_divipola", "anio", "semana_epidemiologica"])
    )

def main():
    frames = []

    for path in FILES:
        if not path.exists():
            print(f"[NO ENCONTRADO] {path}")
            continue

        if path.name.endswith("_998.xlsx"):
            df = process_eda_file(path)
        elif path.name.endswith("_346.xlsx"):
            df = process_346_file(path)
        else:
            continue

        frames.append(df)

    salud = pd.concat(frames, ignore_index=True)

    salud["anio"] = safe_to_numeric(salud["anio"]).astype("Int64")
    salud["semana_epidemiologica"] = safe_to_numeric(salud["semana_epidemiologica"]).astype("Int64")
    salud["casos"] = safe_to_numeric(salud["casos"]).fillna(0)
    salud["cod_divipola"] = salud["cod_divipola"].astype(str).str.zfill(5)

    salud_foco = filter_target_codes(salud)
    salud_foco = fill_target_names(salud_foco)

    salud_foco.to_csv(OUT_DIR / "salud_estandarizada_filas.csv", index=False, encoding="utf-8-sig")

    salud_municipio_semana = aggregate_weekly(salud_foco)
    salud_municipio_semana.to_csv(OUT_DIR / "salud_municipio_semana.csv", index=False, encoding="utf-8-sig")

    eda = salud_municipio_semana[salud_municipio_semana["evento_estandar"] == "EDA"].copy()
    eda.to_csv(OUT_DIR / "eda_municipio_semana.csv", index=False, encoding="utf-8-sig")

    ira_346 = salud_municipio_semana[salud_municipio_semana["evento_estandar"] == "IRA_VIRUS_NUEVO_346"].copy()
    ira_346.to_csv(OUT_DIR / "ira_346_municipio_semana.csv", index=False, encoding="utf-8-sig")

    resumen = (
        salud_municipio_semana.groupby(["evento_estandar", "municipio"], dropna=False, as_index=False)["casos"]
        .sum()
        .sort_values(["evento_estandar", "municipio"])
    )
    resumen.to_csv(OUT_DIR / "resumen_salud_municipios.csv", index=False, encoding="utf-8-sig")

    print("Archivos generados:")
    print("-", OUT_DIR / "salud_estandarizada_filas.csv")
    print("-", OUT_DIR / "salud_municipio_semana.csv")
    print("-", OUT_DIR / "eda_municipio_semana.csv")
    print("-", OUT_DIR / "ira_346_municipio_semana.csv")
    print("-", OUT_DIR / "resumen_salud_municipios.csv")

if __name__ == "__main__":
    main()