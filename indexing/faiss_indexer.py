import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

class FaissIndexer:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.docs: list[str] = []

    def add_documents(self, docs: list[str]):
        cleaned_docs = [doc.strip() for doc in docs if doc.strip()]
        print(f"Документов до очистки: {len(docs)} | После очистки: {len(cleaned_docs)}")
        if not cleaned_docs:
            raise ValueError("Нет непустых документов для индексации.")
        embeddings = self.model.encode(cleaned_docs, convert_to_numpy=True)
        self.index.add(embeddings.astype('float32'))
        self.docs.extend(cleaned_docs)

    def save(self, path: str = '../indexes/faiss.index'):
        faiss.write_index(self.index, path)
        docs_path = os.path.splitext(path)[0] + '.pkl'
        with open(docs_path, 'wb') as f:
            pickle.dump(self.docs, f)

    def load(self, path: str = '../indexes/faiss.index'):
        self.index = faiss.read_index(path)
        docs_path = os.path.splitext(path)[0] + '.pkl'
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                self.docs = pickle.load(f)
        else:
            print(f"[WARN] Не найден файл документов: {docs_path}")
            self.docs = []
