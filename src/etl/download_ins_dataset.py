"""
SentinelaIA Nariño — Descarga automática de microdatos Sivigila (INS)
Descarga IRA (995) + EDA (998) para los años 2023, 2024 y 2025.

Ejecutar desde BI_Health_Nari-o/:
    python src/etl/download_ins_dataset.py

Requiere: selenium, requests
    pip install selenium requests
"""

from pathlib import Path
import time
import requests
import csv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

# ─── PATHS ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "salud"
REPORTS_DIR = BASE_DIR / "data" / "processed" / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIG ────────────────────────────────────────────

DOWNLOAD_TARGETS = [
    ("Morbilidad por IRA", "2023", "IRA_2023_995.xlsx"),
    ("Morbilidad por IRA", "2024", "IRA_2024_995.xlsx"),
    ("Morbilidad por IRA", "2025", "IRA_2025_995.xlsx"),
    ("Morbilidad por EDA", "2023", "EDA_2023_998.xlsx"),
    ("Morbilidad por EDA", "2024", "EDA_2024_998.xlsx"),
    ("Morbilidad por EDA", "2025", "EDA_2025_998.xlsx"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 SentinelaIA"
}

possible_browsers = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    str(Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe"),
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# ─── UTILIDADES ────────────────────────────────────────

def get_browser_path():
    for path in possible_browsers:
        if Path(path).exists():
            return path
    raise FileNotFoundError("No se encontró Chrome ni Edge instalado.")

def clean_text(text):
    return " ".join((text or "").split()).strip()

def download_file(url, output_path):
    response = requests.get(url, headers=HEADERS, stream=True, timeout=120)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(1024 * 256):
            if chunk:
                f.write(chunk)

def save_report(row):
    report_file = REPORTS_DIR / "ins_download_log.csv"
    exists = report_file.exists()

    with open(report_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_name", "year", "file", "url", "status"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ─── SELENIUM ──────────────────────────────────────────

def click_buscar(driver):
    for el in driver.find_elements(By.XPATH, "//*[contains(text(),'Buscar')]"):
        try:
            if el.is_displayed():
                driver.execute_script("arguments[0].click();", el)
                return True
        except:
            pass
    return False

def find_download_link(driver):
    for el in driver.find_elements(By.XPATH, "//a[contains(@href,'.xlsx')]"):
        href = el.get_attribute("href")
        if href:
            return href
    return None

# ─── DESCARGA ──────────────────────────────────────────

def download_one(driver, event_name, year, filename):

    output_path = RAW_DIR / filename

    if output_path.exists():
        print(f"[SKIP] {filename} ya existe")
        return True

    print(f"\nDescargando: {event_name} {year}")

    driver.get("https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx")

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.ID, "lstEvento")))
    wait.until(EC.presence_of_element_located((By.ID, "lstYear")))

    try:
        Select(driver.find_element(By.ID, "lstEvento")).select_by_visible_text(event_name)
        Select(driver.find_element(By.ID, "lstYear")).select_by_visible_text(year)
    except NoSuchElementException:
        print("[ERROR] Evento o año no disponible")
        return False

    time.sleep(2)

    if not click_buscar(driver):
        print("[ERROR] No se pudo hacer click en Buscar")
        return False

    time.sleep(5)

    link = find_download_link(driver)

    if not link:
        print("[ERROR] No se encontró link de descarga")
        return False

    download_file(link, output_path)

    print(f"[OK] {filename} descargado")

    save_report({
        "event_name": event_name,
        "year": year,
        "file": filename,
        "url": link,
        "status": "OK"
    })

    return True

# ─── MAIN ──────────────────────────────────────────────

def main():

    print("="*50)
    print("Descarga INS Sivigila")
    print("="*50)

    browser_path = get_browser_path()

    options = Options()
    options.binary_location = browser_path

    driver = webdriver.Chrome(options=options)

    try:
        for event_name, year, filename in DOWNLOAD_TARGETS:
            download_one(driver, event_name, year, filename)
            time.sleep(2)

    finally:
        driver.quit()

    print("\nProceso terminado")

if __name__ == "__main__":
    main()
