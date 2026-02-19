import os
from datetime import datetime

from scraper import IbexScraper
from market_data import fetch_market_data
from investment_model import run_investment_model
from generate_portfolios import generate_portfolios


def main():
    print("=== IBEXAI weekly job ===")
    print(f"Inicio: {datetime.now().isoformat(timespec='seconds')}")

    print("\n[1/4] Scraping noticias de la última semana...")
    scraper = IbexScraper(save_raw_text=False)
    scraper.run_recent()

    print("\n[2/4] Descargando datos de mercado...")
    fetch_market_data()

    print("\n[3/4] Ejecutando modelo de inversión...")
    run_investment_model()

    print("\n[4/4] Generando carteras...")
    generate_portfolios()
    print("\nFicheros generados:")
    base_dir = os.getcwd()
    print(f"  - {os.path.join(base_dir, 'investment_report.csv')}")
    print(f"  - {os.path.join(base_dir, 'ibex35_portfolios.csv')}")

    print(f"Fin: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
