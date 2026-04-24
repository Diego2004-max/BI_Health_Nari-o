from pathlib import Path
import time
import requests
import csv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "salud"
REPORTS_DIR = BASE_DIR / "data" / "processed" / "reports"
RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EVENT_NAME = "Morbilidad por IRA"
YEAR_VALUE = "2024"

OUTPUT_NAME_MAP = {
    ("Morbilidad por IRA", "2023"): "IRA_2023_995.xlsx",
    ("Morbilidad por IRA", "2024"): "IRA_2024_995.xlsx",
    ("Morbilidad por IRA", "2025"): "IRA_2025_995.xlsx",
    ("Morbilidad por EDA", "2023"): "EDA_2023_998.xlsx",
    ("Morbilidad por EDA", "2024"): "EDA_2024_998.xlsx",
    ("Morbilidad por EDA", "2025"): "EDA_2025_998.xlsx",
}

possible_browsers = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    str(Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe"),
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

browser_path = None
for path in possible_browsers:
    if Path(path).exists():
        browser_path = path
        break

if browser_path is None:
    raise FileNotFoundError("No se encontró Chrome ni Edge instalado.")

print(f"Navegador encontrado: {browser_path}")

options = Options()
options.binary_location = browser_path

driver = webdriver.Chrome(options=options)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SentinelaIA-Narino/1.0"
}

def clean_text(text):
    return " ".join((text or "").split()).strip()

def download_file(url: str, output_path: Path):
    response = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

def save_report(row: dict):
    report_file = REPORTS_DIR / "ins_download_log.csv"
    exists = report_file.exists()
    with open(report_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_name",
                "year_value",
                "result_text",
                "result_href",
                "output_file",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def click_buscar():
    selectors = [
        (By.XPATH, "//button[contains(normalize-space(.), 'Buscar')]"),
        (By.XPATH, "//input[contains(@value, 'Buscar')]"),
        (By.XPATH, "//*[contains(normalize-space(.), 'Buscar')]"),
    ]
    for by, selector in selectors:
        elements = driver.find_elements(by, selector)
        for el in elements:
            try:
                if el.is_displayed():
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", el)
                    return True
            except Exception:
                continue
    return False

def dump_debug(stage_name: str):
    html_file = REPORTS_DIR / f"ins_debug_{stage_name}.html"
    screenshot_file = REPORTS_DIR / f"ins_debug_{stage_name}.png"
    links_file = REPORTS_DIR / f"ins_debug_{stage_name}_links.csv"

    html_file.write_text(driver.page_source, encoding="utf-8")
    driver.save_screenshot(str(screenshot_file))

    rows = []
    for i, a in enumerate(driver.find_elements(By.TAG_NAME, "a"), start=1):
        rows.append({
            "indice": i,
            "text": clean_text(a.text),
            "href": a.get_attribute("href") or "",
            "title": a.get_attribute("title") or "",
            "class": a.get_attribute("class") or "",
        })

    with open(links_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["indice", "text", "href", "title", "class"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"HTML debug guardado en: {html_file}")
    print(f"Screenshot debug guardado en: {screenshot_file}")
    print(f"Links debug guardados en: {links_file}")

    print("\nPrimeros links visibles después de buscar:")
    shown = 0
    for row in rows:
        if row["text"] or row["href"]:
            print(f"- text={row['text']!r} | href={row['href']!r}")
            shown += 1
        if shown >= 20:
            break

def find_result_link():
    candidate_xpaths = [
        "//a[contains(@href, '.xlsx')]",
        "//a[contains(translate(normalize-space(.), 'XLSX', 'xlsx'), 'xlsx')]",
        "//a[contains(., 'Datos_')]",
        "//*[contains(normalize-space(.), 'Datos_')]/ancestor::a[1]",
        "//*[contains(normalize-space(.), 'Formato')]/following::a[1]",
    ]

    for xpath in candidate_xpaths:
        elements = driver.find_elements(By.XPATH, xpath)
        for el in elements:
            try:
                text = clean_text(el.text)
                href = el.get_attribute("href") or ""
                if href:
                    return text, href
            except Exception:
                continue
    return None, None

try:
    driver.get("https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx")
    print("Portal abierto:", driver.current_url)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.ID, "lstEvento")))
    wait.until(EC.presence_of_element_located((By.ID, "lstYear")))

    evento_select = Select(driver.find_element(By.ID, "lstEvento"))
    year_select = Select(driver.find_element(By.ID, "lstYear"))

    evento_select.select_by_visible_text(EVENT_NAME)
    year_select.select_by_visible_text(YEAR_VALUE)

    print(f"Evento seleccionado: {EVENT_NAME}")
    print(f"Año seleccionado: {YEAR_VALUE}")

    time.sleep(2)

    ok = click_buscar()
    if not ok:
        raise RuntimeError("No se pudo hacer clic en el botón Buscar.")

    print("Clic en Buscar realizado.")
    time.sleep(5)

    page_text = clean_text(driver.find_element(By.TAG_NAME, "body").text)
    print("\nFragmento de texto de la página después de buscar:")
    print(page_text[:800])

    result_text, result_href = find_result_link()

    if not result_href:
        dump_debug("after_search")
        raise ValueError("No se encontró href del resultado. Revisa los archivos de debug generados.")

    print("\nResultado encontrado:")
    print("Texto:", result_text)
    print("Href :", result_href)

    output_name = OUTPUT_NAME_MAP.get((EVENT_NAME, YEAR_VALUE))
    if output_name is None:
        safe_event = EVENT_NAME.lower().replace(" ", "_")
        output_name = f"{safe_event}_{YEAR_VALUE}.xlsx"

    output_path = RAW_DIR / output_name
    download_file(result_href, output_path)

    print(f"\nArchivo descargado en: {output_path}")

    save_report({
        "event_name": EVENT_NAME,
        "year_value": YEAR_VALUE,
        "result_text": result_text,
        "result_href": result_href,
        "output_file": str(output_path),
    })

    print(f"Log actualizado en: {REPORTS_DIR / 'ins_download_log.csv'}")

finally:
    driver.quit()