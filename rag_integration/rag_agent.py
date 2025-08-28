import os
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
# Добавим немного типизации для ясности
from typing import List, Tuple, Optional
# ----- НОВЫЕ ИМПОРТЫ -----
from langchain_community.utilities import GoogleSerperAPIWrapper # Используем обертку Serper
from dotenv import load_dotenv # Чтобы убедиться, что ключ загружен

# Загружаем переменные окружения еще раз на всякий случай, если класс импортируется отдельно
load_dotenv()
# ----- КОНЕЦ НОВЫХ ИМПОРТОВ -----


# Предполагаем, что твой self.indexer имеет атрибут .index (объект Faiss)
# и атрибут .docs (список строк/документов)

class RAGAgent:
    def __init__(self, indexer, embed_model_name: str, llm_model_name: str, hf_token: str, top_k: int = 5):
        """
        Инициализирует RAG-агента.

        Args:
            indexer: Объект, содержащий Faiss-индекс (.index) и список документов (.docs).
            embed_model_name: Имя модели для эмбеддингов (SentenceTransformer).
            llm_model_name: Имя модели LLM на Hugging Face Hub (repo_id).
            hf_token: API токен Hugging Face Hub.
            top_k: Количество ближайших документов для извлечения из ЛОКАЛЬНОЙ базы.
        """
        self.indexer = indexer
        self.embed_model_name = embed_model_name
        print(f"Загрузка модели эмбеддингов: {embed_model_name}...")
        self.embedder = SentenceTransformer(embed_model_name)
        print("Модель эмбеддингов загружена.")
        self.top_k = top_k

        print(f"Инициализация LLM Endpoint: {llm_model_name}...")
        self.llm = HuggingFaceEndpoint(
            repo_id=llm_model_name,
            task="text-generation", # Оставляем text-generation, как рекомендовано
            huggingfacehub_api_token=hf_token,
            # Добавим параметры для контроля генерации
            max_new_tokens=1024,  # Увеличим лимит ответа еще немного для комбинированного контекста
            temperature=0.6,     # Можно еще чуть уменьшить для большей фактологичности
            # repetition_penalty=1.1 # Опционально, чтобы уменьшить повторы
        )
        print("LLM Endpoint инициализирован.")

        # Проверка наличия необходимых атрибутов у indexer
        if not hasattr(self.indexer, 'index') or not hasattr(self.indexer, 'docs'):
            raise ValueError("Объект indexer должен иметь атрибуты 'index' (Faiss индекс) и 'docs' (список строк).")

        # ----- ИНИЦИАЛИЗАЦИЯ SERPER WRAPPER -----
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            print("ПРЕДУПРЕЖДЕНИЕ: SERPER_API_KEY не найден в переменных окружения. Веб-поиск будет недоступен.")
            self.search_wrapper = None
        else:
            print("Инициализация обертки Serper API...")
            try:
                # Можно настроить k (количество результатов), gl (страна), hl (язык)
                # Для русского поиска можно попробовать: gl='ru', hl='ru'
                self.search_wrapper = GoogleSerperAPIWrapper(k=4, gl='ru', hl='ru', serper_api_key=serper_api_key) # Ищем 4 результата на русском
                print("Обертка Serper API инициализирована.")
            except Exception as e:
                print(f"Ошибка инициализации Serper API Wrapper: {e}")
                self.search_wrapper = None
        # ----- КОНЕЦ ИНИЦИАЛИЗАЦИИ SERPER WRAPPER -----

    def ask(self, query: str) -> str:
        """
        Выполняет RAG-пайплайн: веб-поиск -> поиск в базе -> сборка контекста -> запрос к LLM.

        Args:
            query: Вопрос пользователя.

        Returns:
            Ответ от LLM, основанный на найденном контексте, очищенный от мусора.
        """
        print(f"\n[RAG Agent] Получен запрос: '{query}'")

        # --- 0. Веб-Поиск (Serper) ---
        web_results_text = ""
        if self.search_wrapper:
            try:
                print("[RAG Agent] Выполняю веб-поиск через Serper...")
                # Используем search_wrapper.run() для получения текстового резюме
                web_results_text = self.search_wrapper.run(query)
                # Можно попробовать получить больше деталей через .results() и собрать сниппеты
                # web_results_dict = self.search_wrapper.results(query)
                # snippets = [f"Источник: {r.get('link', 'N/A')}\n{r.get('snippet', '')}" for r in web_results_dict.get('organic', []) if r.get('snippet')]
                # if snippets:
                #    web_results_text = "\n\n".join(snippets)
                # else:
                #    web_results_text = web_results_dict.get('answerBox', {}).get('answer', '') # Попробуем answer box
                print(f"[RAG Agent] Результаты веб-поиска получены (длина: {len(web_results_text)}).")
            except Exception as e:
                print(f"[RAG Agent] Ошибка веб-поиска Serper: {e}")
                web_results_text = "" # Продолжаем без веб-результатов
        else:
            print("[RAG Agent] Веб-поиск пропущен (Serper API не настроен).")


        # --- 1. Поиск в Локальной Базе (Faiss Retrieval) ---
        local_context = ""
        context_docs = []
        valid_indexes = []
        try:
            print(f"[RAG Agent] Кодирую запрос с помощью {self.embed_model_name}...")
            query_vec = self.embedder.encode([query], normalize_embeddings=True).astype('float32')
            print(f"[RAG Agent] Выполняю поиск top-{self.top_k} документов в локальной базе...")
            D, I = self.indexer.index.search(query_vec, self.top_k)

            if len(I) > 0 and len(I[0]) > 0:
                potential_indexes = I[0]
                num_docs = len(self.indexer.docs)
                valid_indexes = [i for i in potential_indexes if 0 <= i < num_docs]
                if valid_indexes:
                    print(f"[RAG Agent] Найдено {len(valid_indexes)} релевантных локальных документов с индексами: {valid_indexes}")
                    context_docs = [self.indexer.docs[i] for i in valid_indexes]
                    local_context = "\n\n---\n\n".join(context_docs) # Разделяем документы
                else:
                    print(f"[RAG Agent] Локальные индексы ({potential_indexes}) выходят за пределы диапазона.")
            else:
                print("[RAG Agent] Локальный поиск не вернул результатов.")

        except Exception as e:
            print(f"[RAG Agent] Ошибка во время локального кодирования или поиска: {e}")
            # Не прерываем выполнение, можем использовать только веб-поиск


        # --- Определим стандартную фразу-отказ (на русском), обновленную ---
        refusal_phrase_ru = "В предоставленных данных (включая веб-поиск) нет информации по этому вопросу."

        # --- 2. Сборка Итогового Контекста ---
        print("[RAG Agent] Собираю итоговый контекст...")
        final_context = ""
        context_parts = [] # Собираем части, чтобы потом соединить
        if web_results_text and web_results_text.strip(): # Проверяем, что веб-результат не пустой
            context_parts.append(f"===== Информация из Веб-Поиска =====\n{web_results_text.strip()}")
        if local_context and local_context.strip(): # Проверяем, что локальный контекст не пустой
            context_parts.append(f"===== Информация из Базы Новостей =====\n{local_context.strip()}")

        # Соединяем части, если они есть
        final_context = "\n\n---\n\n".join(context_parts)

        if not final_context.strip(): # Если контекст все еще пуст
            print("[RAG Agent] Не найдено релевантной информации ни в базе, ни в вебе.")
            return refusal_phrase_ru # Возвращаем отказ

        # Ограничение длины итогового контекста
        max_context_length = 15000 # Уменьшим немного, т.к. сам промпт тоже занимает место
        if len(final_context) > max_context_length:
            print(f"[RAG Agent] Итоговый контекст слишком длинный ({len(final_context)} символов), обрезается до {max_context_length}.")
            final_context = final_context[:max_context_length] + "..."


        # --- 3. Формирование Промпта (Prompt Engineering) ---
        # !!!!! ИСПОЛЬЗУЕМ УСИЛЕННЫЙ ПРОМПТ И FINAL_CONTEXT !!!!!
        prompt = f"""**Инструкция:** Проанализируй следующий контекст (который может включать информацию из веб-поиска и/или локальной базы новостей). Затем ответь на вопрос пользователя **строго на русском языке**. Твой ответ должен быть основан **исключительно** на информации из предоставленного контекста. Не добавляй информацию, которой нет в тексте. Не выдумывай факты. Если информация для ответа полностью отсутствует в предоставленном контексте, напиши **только** фразу **на русском языке**: "{refusal_phrase_ru}"

**Контекст:**
{final_context}

**Вопрос пользователя:** {query}

**Ответ (на русском языке):**"""


        # --- 4. Запрос к LLM (Generation) ---
        print("[RAG Agent] Отправляю запрос к LLM...")
        try:
            response = self.llm.invoke(prompt)
            print("[RAG Agent] Ответ от LLM получен.")
        except Exception as e:
            print(f"[RAG Agent] Ошибка при вызове LLM ({self.llm.repo_id}): {e}")
            return f"Произошла ошибка при обращении к языковой модели: {e}"

        # --- 5. Постобработка Ответа ---
        # !!!!! ИСПОЛЬЗУЕМ ЛОГИКУ ПОСТОБРАБОТКИ !!!!!
        response = response.strip()
        cleaned_response = response

        extra_output_markers = ["---", "### Пример:", "**Инструкция:**", "**Контекст:**", "**Вопрос:**", "**Ответ (на русском языке):**"] # Добавим еще один возможный маркер
        refusal_phrase_en = "There is no information on this matter in the provided news." # Старый отказ

        is_refusal = cleaned_response.startswith(refusal_phrase_ru) or cleaned_response.startswith(refusal_phrase_en)

        for marker in extra_output_markers:
            marker_pos = cleaned_response.find(marker)
            if marker_pos > 5:
                if is_refusal and marker_pos < max(len(refusal_phrase_ru), len(refusal_phrase_en)) + 5:
                    continue
                cleaned_response = cleaned_response[:marker_pos].strip()
                print(f"[RAG Agent] Обнаружен маркер '{marker}', ответ обрезан.")
                break

        if not cleaned_response.strip():
            print("[RAG Agent] Ответ стал пустым после очистки.")
            cleaned_response = refusal_phrase_ru
        elif is_refusal and cleaned_response != refusal_phrase_ru and cleaned_response != refusal_phrase_en :
            print("[RAG Agent] Восстановлен стандартный ответ-отказ после некорректной обрезки.")
            cleaned_response = refusal_phrase_ru

        return cleaned_response.strip() # Возвращаем очищенный ответ