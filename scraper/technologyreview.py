import os
# Импортируем НОВУЮ функцию из обновленного base_scrape
from scraper.base_scraper import scrape_articles_from_site
from yaml import safe_load

# Корневая директория проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config.yaml')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Чтение конфига
with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = safe_load(f)

def scrape_technologyreview_ai():
    """Скрейпит статьи с MIT Technology Review AI с использованием сайтмапа."""
    print("\n--- Starting MIT Technology Review AI Scraping ---")
    # Ищем конфигурацию для Technology Review в YAML
    site_config = next((site for site in cfg.get('scraping', {}).get('sites', []) if site.get('name') == 'MIT Technology Review AI'), None) # Убедись, что 'name' в YAML совпадает

    if not site_config:
        print("Error: Configuration for 'MIT Technology Review AI' not found in config.yaml.")
        return

    sitemap_url = site_config.get('sitemap_url')
    if not sitemap_url:
        print("Error: 'sitemap_url' not found for MIT Technology Review AI in config.yaml.")
        return

    # Получаем лимиты и задержки
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
    print("--- Finished MIT Technology Review AI Scraping ---")


if __name__ == '__main__':
    scrape_technologyreview_ai()