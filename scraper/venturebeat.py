import os
from scraper.base_scraper import scrape_articles_from_site
from yaml import safe_load

# Корневая директория проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config.yaml')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Чтение конфига
with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = safe_load(f)

def scrape_venturebeat_ai():
    """Скрейпит статьи с VentureBeat AI с использованием сайтмапа."""
    print("\n--- Starting VentureBeat AI Scraping ---")
    # Ищем конфигурацию для VentureBeat в YAML (используем 'name' для поиска)
    site_config = next((site for site in cfg.get('scraping', {}).get('sites', []) if site.get('name') == 'VentureBeat AI'), None)

    if not site_config:
        print("Error: Configuration for 'VentureBeat AI' not found in config.yaml.")
        return

    sitemap_url = site_config.get('sitemap_url')
    if not sitemap_url:
        print("Error: 'sitemap_url' not found for VentureBeat AI in config.yaml.")
        return

    # Получаем лимиты и задержки из конфига или используем значения по умолчанию
    limit = site_config.get('limit', cfg.get('scraping', {}).get('article_limit_per_site', 50))
    delay = site_config.get('delay', cfg.get('scraping', {}).get('delay_between_articles', 2))

    # Вызываем НОВУЮ функцию
    scrape_articles_from_site(
        output_dir=RAW_DIR,
        sitemap_url=sitemap_url,
        delay_between_articles=delay,
        limit=limit,
        max_sitemaps=15 
    )
    print("--- Finished VentureBeat AI Scraping ---")


if __name__ == '__main__':
    scrape_venturebeat_ai()