import os
import time
import pandas as pd
import newspaper
from newspaper import Article
from ddgs import DDGS
from tqdm import tqdm
import nltk
from companies import IBEX35_COMPANIES, IBEX35_SECTORS, TARGET_DOMAINS
from datetime import datetime
import logging
import random
import htmldate
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from pysentimiento import create_analyzer
import torch

logging.basicConfig(
    filename='scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class IbexScraper:
    def __init__(self, output_file='ibex35_news_sentiment.csv', save_raw_text=True):
        self.output_file = output_file
        self.save_raw_text = save_raw_text
        self.columns = [
            'date', 'url', 'title', 'ticker', 'company', 'sector', 'scope',
            'sentiment_label', 'sentiment_score', 'prob_pos', 'prob_neg', 'prob_neu',
            'calibrated_score'
        ]
        if self.save_raw_text:
            self.columns.insert(3, 'text')
        
        print("Cargando modelo de sentimientos (pysentimiento)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")
        
        try:
            self.analyzer = create_analyzer(task="sentiment", lang="es")
        except Exception as e:
            logging.error(f"Error cargando pysentimiento: {e}")
            self.analyzer = None

        if os.path.exists(self.output_file):
            self.df = pd.read_csv(self.output_file)
            self.scraped_urls = set(self.df['url'].tolist())
            
            missing_cols = [c for c in self.columns if c not in self.df.columns]
            if missing_cols:
                print(f"Migrando CSV, añadiendo columnas: {missing_cols}")
                for c in missing_cols:
                    self.df[c] = pd.NA
            
            self.df = self.df[self.columns]
            self.df.to_csv(self.output_file, index=False, encoding='utf-8')
        else:
            self.df = pd.DataFrame(columns=self.columns)
            self.scraped_urls = set()

    def search_urls(self, query, limit=10, timelimit=None):
        logging.info(f"Buscando en DDG: {query} (limit={limit}, time={timelimit})")
        links = []
        
        ddg = DDGS()
        try:
            results = ddg.text(query, region="es-es", max_results=limit, timelimit=timelimit)
            if results:
                for r in results:
                    links.append(r['href'])
            else:
                logging.warning(f"No resultados para {query}")
                
        except Exception as e:
            logging.error(f"Error DDG {query}: {e}")
            time.sleep(random.uniform(2, 5))
        
        domain_limit = max(1, min(3, limit))
        for domain in TARGET_DOMAINS:
            site_query = f"site:{domain} {query}"
            try:
                results = ddg.text(site_query, region="es-es", max_results=domain_limit, timelimit=timelimit)
                if results:
                    for r in results:
                        links.append(r['href'])
            except Exception as e:
                logging.error(f"Error DDG site:{domain} {query}: {e}")
                time.sleep(random.uniform(1, 3))

        return list(set(links))

    def download_with_cffi(self, url):
        try:
            response = cffi_requests.get(url, impersonate="chrome", timeout=15)
            return response
        except Exception as e:
            logging.error(f"Error CFFI {url}: {e}")
            return None

    def extract_pdf_content(self, content):
        text = ""
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            logging.error(f"Error extrayendo PDF: {e}")
        return text

    def extract_content(self, url, year, context_data):
        response = self.download_with_cffi(url)
        if not response or response.status_code != 200:
            return None
            
        title = None
        text = None
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            text = self.extract_pdf_content(response.content)
            title = url.split('/')[-1].replace('.pdf', '').replace('-', ' ')
            pub_date_str = f"01/01/{year}"
        else:
            try:
                article = Article(url)
                article.set_html(response.text)
                article.parse()
                try:
                    article.nlp()
                except:
                    pass
                title = article.title
                text = article.text
                
                if not title:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    if soup.title:
                        title = soup.title.get_text().strip()
            except Exception as e:
                logging.error(f"Error parsing HTML {url}: {e}")

        if not title and not text:
            return None

        if not title:
            title = url.split('/')[-1].replace('-', ' ')

        pub_date_str = None
        try:
            if not 'application/pdf' in content_type:
                found_date = htmldate.find_date(response.text, url=url, outputformat='%d/%m/%Y')
                if found_date:
                    pub_date_str = found_date
        except:
            pass
        
        if not pub_date_str:
            pub_date_str = f"01/01/{year}"

        return {
            'date': pub_date_str,
            'url': url,
            'title': title,
            'text': text or "",
            **context_data
        }

    def analyze_batch_sentiment(self, articles):
        if not self.analyzer:
            return articles

        print(f"    Analizando sentimientos de {len(articles)} artículos...")
        for article in tqdm(articles, desc="    Sentiment", leave=False):
            text_full = article['text']
            if not text_full or len(text_full) < 50:
                text_full = article['title'] or ""
            
            chunk_size = 1500
            chunks = [text_full[i:i+chunk_size] for i in range(0, len(text_full), chunk_size)]
            
            if not chunks:
                chunks = [""]

            pos_sum, neg_sum, neu_sum = 0.0, 0.0, 0.0
            valid_chunks = 0
            
            try:
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    res = self.analyzer.predict(chunk)
                    pos_sum += res.probas.get('POS', 0.0)
                    neg_sum += res.probas.get('NEG', 0.0)
                    neu_sum += res.probas.get('NEU', 0.0)
                    valid_chunks += 1
                
                if valid_chunks > 0:
                    prob_pos = pos_sum / valid_chunks
                    prob_neg = neg_sum / valid_chunks
                    prob_neu = neu_sum / valid_chunks
                else:
                    prob_pos, prob_neg, prob_neu = 0.0, 0.0, 1.0

                if prob_pos > prob_neg and prob_pos > prob_neu:
                    label = 'POS'
                    score = prob_pos
                elif prob_neg > prob_pos and prob_neg > prob_neu:
                    label = 'NEG'
                    score = prob_neg
                else:
                    label = 'NEU'
                    score = prob_neu

                article.update({
                    'sentiment_label': label,
                    'sentiment_score': score,
                    'prob_pos': prob_pos,
                    'prob_neg': prob_neg,
                    'prob_neu': prob_neu,
                    'calibrated_score': prob_pos - prob_neg
                })
            except Exception as e:
                logging.error(f"Error análisis sentimiento: {e}")
                article.update({
                    'sentiment_label': 'NEU', 'sentiment_score': 0.0,
                    'prob_pos': 0.0, 'prob_neg': 0.0, 'prob_neu': 1.0,
                    'calibrated_score': 0.0
                })
        return articles

    def save_batch(self, new_articles):
        if not new_articles:
            return

        new_df = pd.DataFrame(new_articles)
        for col in self.columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA
        new_df = new_df[self.columns]

        if not os.path.exists(self.output_file):
            new_df.to_csv(self.output_file, mode='w', header=True, index=False, encoding='utf-8')
        else:
            new_df.to_csv(self.output_file, mode='a', header=False, index=False, encoding='utf-8')
            
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.scraped_urls.update(new_df['url'].tolist())
        logging.info(f"Guardados {len(new_articles)} artículos nuevos.")

    def process_query_group(self, query_list, year, max_workers, limit_per_query=20, timelimit=None):
        for query_str, context in query_list:
            links = self.search_urls(query_str, limit=limit_per_query, timelimit=timelimit)
            if not links:
                continue

            links = [l for l in links if l not in self.scraped_urls]
            if not links:
                continue

            print(f"  Query: '{query_str}' -> {len(links)} nuevos links")

            new_articles = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.extract_content, url, year, context): url for url in links}
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        new_articles.append(res)
            
            if new_articles:
                new_articles = self.analyze_batch_sentiment(new_articles)
                self.save_batch(new_articles)
            
            time.sleep(random.uniform(1.0, 2.0))

    def run_recent(self):
        print(f"Iniciando scraping de noticias RECIENTES (última semana)...")
        current_year = datetime.now().year
        
        print("\n=== MACRO & MERCADO (Última Semana) ===")
        macro_queries = [
            (f"economia españa actualidad", {'scope': 'national', 'ticker': 'ESP', 'sector': 'Macro'}),
            (f"ibex35 analisis semanal", {'scope': 'national', 'ticker': 'IBEX', 'sector': 'Market'}),
            (f"banco central europeo tipos interes", {'scope': 'international', 'ticker': 'ECB', 'sector': 'Macro'}),
        ]
        self.process_query_group(macro_queries, current_year, max_workers=5, limit_per_query=5, timelimit='w')
        
        print("\n=== EMPRESAS IBEX35 (Última Semana) ===")
        company_queries = []
        for ticker, name in IBEX35_COMPANIES.items():
            sec = IBEX35_SECTORS.get(ticker, 'Unknown')
            q = f'"{name}" noticias finanzas economia'
            company_queries.append((q, {'scope': 'company', 'ticker': ticker, 'company': name, 'sector': sec}))
        
        chunk_size = 5
        for i in range(0, len(company_queries), chunk_size):
            chunk = company_queries[i:i+chunk_size]
            self.process_query_group(chunk, current_year, max_workers=10, limit_per_query=12, timelimit='w')

    def run(self, start_year=2026, end_year=2015):
        print(f"Iniciando scraping unificado ({start_year}-{end_year})...")
        years = range(start_year, end_year - 1, -1)
        
        for year in years:
            print(f"\n=== AÑO {year} ===")
            
            print("--- Scope: International ---")
            intl_queries = [
                (f"economy usa {year}", {'scope': 'international', 'ticker': 'SPX', 'sector': 'Macro'}),
                (f"federal reserve interest rates {year}", {'scope': 'international', 'ticker': 'FED', 'sector': 'Macro'}),
                (f"eurozone economy {year}", {'scope': 'international', 'ticker': 'EUR', 'sector': 'Macro'}),
                (f"ecb interest rates {year}", {'scope': 'international', 'ticker': 'ECB', 'sector': 'Macro'}),
                (f"oil prices forecast {year}", {'scope': 'international', 'ticker': 'OIL', 'sector': 'Energy'}),
            ]
            self.process_query_group(intl_queries, year, max_workers=5, limit_per_query=20)

            print("--- Scope: National (Spain) ---")
            nat_queries = [
                (f"economia españa {year}", {'scope': 'national', 'ticker': 'ESP', 'sector': 'Macro'}),
                (f"ibex35 prevision {year}", {'scope': 'national', 'ticker': 'IBEX', 'sector': 'Market'}),
                (f"prima de riesgo españa {year}", {'scope': 'national', 'ticker': 'ESP', 'sector': 'Macro'}),
                (f"banco de españa informe {year}", {'scope': 'national', 'ticker': 'BDE', 'sector': 'Macro'}),
            ]
            self.process_query_group(nat_queries, year, max_workers=5, limit_per_query=20)

            print("--- Scope: Sectors ---")
            unique_sectors = set(IBEX35_SECTORS.values())
            sector_queries = []
            for sec in unique_sectors:
                query = f"sector {sec} españa {year}"
                sector_queries.append((query, {'scope': 'sector', 'ticker': None, 'sector': sec}))
            
            self.process_query_group(sector_queries, year, max_workers=5, limit_per_query=20)

            print("--- Scope: Companies ---")
            company_queries = []
            for ticker, name in IBEX35_COMPANIES.items():
                sec = IBEX35_SECTORS.get(ticker, 'Unknown')
                queries_variations = [
                    f'"{name}" resultados financieros {year}',
                    f'"{name}" acciones bolsa {year}',
                    f'"{name}" noticias economia {year}',
                    f'"{name}" estrategia inversion {year}',
                    f'"{name}" {year}'
                ]
                for q in queries_variations:
                    company_queries.append((q, {'scope': 'company', 'ticker': ticker, 'company': name, 'sector': sec}))
            
            chunk_size = 10 
            for i in range(0, len(company_queries), chunk_size):
                chunk = company_queries[i:i+chunk_size]
                self.process_query_group(chunk, year, max_workers=10, limit_per_query=50)

if __name__ == "__main__":
    scraper = IbexScraper(save_raw_text=False)
    scraper.run_recent()
