import argparse
import os
from dotenv import load_dotenv
from yaml import safe_load
from scraper.venturebeat import scrape_venturebeat_ai
from scraper.technologyreview import scrape_technologyreview_ai
from preprocessing.cleaner import clean_text
from preprocessing.chunker import chunk_text
from indexing.faiss_indexer import FaissIndexer
from rag_integration.rag_agent import RAGAgent
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

# Загрузка конфига
BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, 'configs/config.yaml'), encoding='utf-8') as f:
    cfg = safe_load(f)

DATA_RAW = os.path.join(BASE_DIR, 'data/raw')
DATA_PROC = os.path.join(BASE_DIR, 'data/processed')
INDEX_PATH = os.path.join(BASE_DIR, 'indexes/faiss.index')


def step_scrape():
    """Запускает скрейпинг для сайтов, определенных в коде."""
    print("--- Starting Scrape Step ---")
    # Вызываем обновленные функции скрейпинга
    scrape_venturebeat_ai()
    scrape_technologyreview_ai()
    # Добавь сюда вызовы для других сайтов, если ты создашь для них скреперы
    print("--- Scrape Step Finished ---")


def step_preprocess():
    """Шаг предобработки: очистка текста и chunking с отладочными сообщениями"""
    os.makedirs(DATA_PROC, exist_ok=True)
    total_files = 0
    total_chunks = 0
    for fname in os.listdir(DATA_RAW):
        raw_path = os.path.join(DATA_RAW, fname)
        with open(raw_path, encoding='utf-8') as f:
            text = f.read()
        total_files += 1
        print(f"[DEBUG preprocess] Файл: {fname} | Размер: {len(text)} символов")
        clean = clean_text(text)
        print(f"[DEBUG preprocess] После очистки: {len(clean)} символов")
        chunks = chunk_text(
            clean,
            cfg['rag']['chunk_size'],
            cfg['rag']['chunk_overlap']
        )
        valid_chunks = [c for c in chunks if c.strip()]
        print(f"[DEBUG preprocess] Генерация чанков: всего {len(chunks)}, валидных {len(valid_chunks)}")
        for i, chunk in enumerate(valid_chunks):
            total_chunks += 1
            out_fname = f"{os.path.splitext(fname)[0]}_chunk_{i}.txt"
            out_path = os.path.join(DATA_PROC, out_fname)
            with open(out_path, 'w', encoding='utf-8') as outf:
                outf.write(chunk)
    print(f"[INFO preprocess] Всего файлов: {total_files}, сохранено чанков: {total_chunks}")


def step_index():
    indexer = FaissIndexer()
    docs = []
    for fname in os.listdir(DATA_PROC):
        with open(os.path.join(DATA_PROC, fname), encoding='utf-8') as f:
            docs.append(f.read())
    indexer.add_documents(docs)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    indexer.save(INDEX_PATH)


def step_rag(query: str):
    indexer = FaissIndexer()
    indexer.load(INDEX_PATH)
    embed_model = SentenceTransformer(cfg['rag']['embedding_model_name'])
    agent = RAGAgent(
    indexer=indexer,
    embed_model_name=cfg['rag']['embedding_model_name'],
    llm_model_name=cfg['rag']['llm_model_name'],
    hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    top_k=cfg['rag']['top_k']
    )
    print(agent.ask(query))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['scrape','preprocess','index','rag'], required=True)
    parser.add_argument('--query', type=str, help='Вопрос для RAG')
    args = parser.parse_args()

    if args.step == 'scrape':
        step_scrape()
    elif args.step == 'preprocess':
        step_preprocess()
    elif args.step == 'index':
        step_index()
    elif args.step == 'rag':
        if not args.query:
            parser.error('--query is required for rag step')
        step_rag(args.query)

if __name__ == '__main__':
    main()