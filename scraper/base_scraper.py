import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
from fake_useragent import UserAgent
from newspaper import Article, Config # Импортируем Config для UserAgent
from tqdm import tqdm # Добавим прогресс-бар
import os # Для работы с путями и файлами
import re # Для очистки имен файлов

ua = UserAgent()

def setup_newspaper_config() -> Config:
    """Создает конфигурацию для newspaper3k с User-Agent."""
    config = Config()
    config.browser_user_agent = ua.random
    config.request_timeout = 15
    config.fetch_images = False # Не скачиваем картинки
    config.memoize_articles = False # Не кэшируем статьи в памяти
    return config

def safe_filename(text: str, max_len: int = 100) -> str:
    """Создает безопасное имя файла из строки."""
    # Удаляем недопустимые символы
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Заменяем пробелы на подчеркивания
    text = text.replace(' ', '_')
    # Ограничиваем длину
    return text[:max_len].strip('_')

def extract_and_save_article(url: str, output_dir: str, config: Config) -> bool:
    """
    Скачивает, парсит статью с помощью newspaper3k и сохраняет в файл.

    Args:
        url: URL статьи.
        output_dir: Папка для сохранения (например, data/raw).
        config: Конфигурация newspaper3k.

    Returns:
        True если успешно, False при ошибке.
    """
    try:
        # Убрал дублирующийся print, он есть в scrape_articles_from_site
        # print(f"  [Article] Processing: {url}")
        article = Article(url, config=config)
        article.download()
        # time.sleep(0.2) # Можно добавить минимальную паузу
        article.parse()

        article_text = article.text
        article_title = article.title

        if not article_text or len(article_text) < 100: # Порог текста
            print(f"  [Article] Warning: Not enough text found for {url}. Skipping.")
            return False

        # Создаем имя файла из заголовка или URL
        filename_base = safe_filename(article_title)
        if not filename_base:
            # Если заголовок плохой, берем часть URL
            path_parts = urlparse(url).path.split('/')
            filename_base = safe_filename(path_parts[-1] or path_parts[-2] or f"article_{int(time.time())}")

        filename = os.path.join(output_dir, f"{filename_base}.txt")

        # Сохраняем
        os.makedirs(output_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Title: {article_title}\n")
            f.write(f"URL: {url}\n\n")
            f.write(article_text)
        # print(f"  [Article] Saved: {filename}") # Можно включить для подробного лога
        return True

    except Exception as e:
        print(f"  [Article] Error processing {url}: {e}")
        return False

# --- НАЧАЛО ЗАМЕНЕННОЙ ФУНКЦИИ ---
def get_article_urls_from_sitemap(initial_sitemap_url: str, limit: int = 50, max_sitemaps_to_check: int = 10) -> list[str]:
    """
    Извлекает URL статей из XML сайтмапа, обрабатывая вложенные индексные сайтмапы.

    Args:
        initial_sitemap_url: Начальный URL сайтмапа (может быть индексным).
        limit: Максимальное количество URL статей для извлечения.
        max_sitemaps_to_check: Ограничение на количество сайтмапов для проверки, чтобы избежать бесконечного обхода.

    Returns:
        Список URL статей.
    """
    sitemaps_to_process = [initial_sitemap_url] # Очередь сайтмапов для проверки
    processed_sitemaps = set()            # Множество уже обработанных сайтмапов
    article_urls_found = set()            # Множество найденных URL статей (уникальные)
    sitemaps_checked_count = 0

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg')
    # Расширим список исключений, включая типичные для индексных сайтмапов
    exclude_paths = ('/category/', '/tag/', '/author/', '/wp-content/uploads/', 'sitemap-index', 'image-sitemap', 'video-sitemap')

    headers = {"User-Agent": ua.random}

    while sitemaps_to_process and sitemaps_checked_count < max_sitemaps_to_check and len(article_urls_found) < limit:
        current_sitemap_url = sitemaps_to_process.pop(0) # Берем первый из очереди

        if current_sitemap_url in processed_sitemaps:
            continue # Пропускаем, если уже обрабатывали

        print(f"[Sitemap] Processing sitemap: {current_sitemap_url}")
        processed_sitemaps.add(current_sitemap_url)
        sitemaps_checked_count += 1

        try:
            response = requests.get(current_sitemap_url, headers=headers, timeout=25)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()

            # Убедимся, что это XML
            if 'xml' not in content_type:
                print(f"[Sitemap] Warning: Skipping non-XML content type '{content_type}' for URL: {current_sitemap_url}")
                continue

            soup = BeautifulSoup(response.content, 'lxml-xml')

            # Проверяем, это индексный сайтмап (<sitemapindex>) или обычный (<urlset>)
            if soup.find('sitemapindex'):
                print(f"[Sitemap] Detected sitemap index. Parsing for child sitemaps...")
                # Ищем ссылки на дочерние сайтмапы
                sitemaps = soup.find_all('sitemap')
                for sitemap in sitemaps:
                    loc = sitemap.find('loc')
                    if loc:
                        child_sitemap_url = loc.text.strip()
                        if child_sitemap_url not in processed_sitemaps:
                            sitemaps_to_process.append(child_sitemap_url)
                            # print(f"  [Sitemap] Added child sitemap to queue: {child_sitemap_url}")

            elif soup.find('urlset'):
                # print(f"[Sitemap] Detected regular sitemap. Parsing for article URLs...")
                # Ищем ссылки на страницы/статьи
                urls = soup.find_all('url')
                for url_entry in urls:
                    loc = url_entry.find('loc')
                    if loc:
                        url = loc.text.strip()
                        path = urlparse(url).path.lower()

                        # Применяем фильтрацию
                        is_image = url.lower().endswith(image_extensions)
                        # Используем немного другую проверку для исключенных путей
                        is_excluded = any(ex_path in url.lower() for ex_path in exclude_paths if ex_path)
                        is_valid_path = path and len(path) > 1

                        if is_valid_path and not is_image and not is_excluded:
                            if len(article_urls_found) < limit:
                                article_urls_found.add(url)
                            else:
                                break # Достигли лимита статей
                    if len(article_urls_found) >= limit:
                        break # Выходим из цикла по url_entry

            else:
                print(f"[Sitemap] Warning: Unknown sitemap format for {current_sitemap_url}. Root tag not <sitemapindex> or <urlset>.")

        except requests.exceptions.RequestException as e:
            print(f"[Sitemap] Error fetching sitemap {current_sitemap_url}: {e}")
        except Exception as e:
            print(f"[Sitemap] Error processing sitemap {current_sitemap_url}: {e}")

        # Небольшая пауза между запросами к сайтмапам
        time.sleep(0.5)


    final_urls = list(article_urls_found)
    print(f"[Sitemap] Finished processing. Extracted {len(final_urls)} unique article URLs after checking {sitemaps_checked_count} sitemaps.")
    return final_urls
# --- КОНЕЦ ЗАМЕНЕННОЙ ФУНКЦИИ ---


# --- НАЧАЛО ЗАМЕНЕННОЙ ФУНКЦИИ ---
def scrape_articles_from_site(
    output_dir: str,
    sitemap_url: str,
    delay_between_articles: int = 1,
    limit: int = 20,
    max_sitemaps: int = 10 # Добавим параметр для передачи в get_article_urls_from_sitemap
):
    """
    Основная функция: получает URL из сайтмапа (обрабатывая вложенные) и скрейпит каждую статью.

    Args:
        output_dir: Папка для сохранения статей (data/raw).
        sitemap_url: URL к sitemap.xml.
        delay_between_articles: Пауза (в секундах) между скачиванием статей.
        limit: Максимальное количество статей для скрейпинга.
        max_sitemaps: Максимальное количество сайтмапов для проверки.
    """
    print(f"\n--- Starting scraping for {sitemap_url} ---")
    # Вызываем обновленную функцию
    article_urls = get_article_urls_from_sitemap(sitemap_url, limit=limit * 2, max_sitemaps_to_check=max_sitemaps) # Запрашиваем чуть больше URL на случай ошибок

    if not article_urls:
        print("No article URLs found or extracted. Stopping.")
        return

    config = setup_newspaper_config()
    success_count = 0
    fail_count = 0

    # Используем итератор, чтобы не обрабатывать лишние URL, если лимит достигнут
    processed_urls = 0
    for url in tqdm(article_urls, desc=f"Scraping articles from {urlparse(sitemap_url).netloc}"):
        processed_urls += 1
        # Обрабатываем только до лимита УСПЕШНЫХ скачиваний
        if success_count >= limit:
            print(f"\nReached target limit of {limit} successfully scraped articles.")
            break # Выходим из цикла for

        print(f"  [Article] Processing URL {processed_urls}/{len(article_urls)}: {url}") # Добавим лог URL
        if extract_and_save_article(url, output_dir, config):
            success_count += 1
        else:
            fail_count += 1

        # !!! ВАЖНАЯ ПАУЗА !!!
        # Применяем паузу только если еще не достигли лимита
        if success_count < limit:
            time.sleep(delay_between_articles)

    print(f"--- Finished scraping for {sitemap_url} ---")
    print(f"Successfully scraped: {success_count}")
    print(f"Failed attempts: {fail_count} (out of {processed_urls} processed URLs)")
# --- КОНЕЦ ЗАМЕНЕННОЙ ФУНКЦИИ ---