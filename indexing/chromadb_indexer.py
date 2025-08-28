import chromadb
from chromadb.utils import embedding_functions
from yaml import safe_load
import os

# Пример ChromaDB клиента
client = chromadb.Client()

def build_chroma_collection(collection_name: str, docs: list[str]):
    # Используем OpenAI эмбеддер
    from chromadb.config import Settings
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')
    embed = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed
    )
    collection.add(documents=docs)