from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

class RequestBody(BaseModel):
    title: str
    topics: list

# Инициализация BERT модели
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def find_closest_topic(title, topics):
    # Получение эмбеддингов заголовка и тем
    title_embedding = model.encode(title, convert_to_tensor=True)
    topics_embeddings = model.encode(topics, convert_to_tensor=True)

    # Перенос тензоров на CPU и конвертация в numpy массивы
    title_embedding = title_embedding.cpu().numpy()
    topics_embeddings = topics_embeddings.cpu().numpy()

    # Вычисление косинусного сходства
    cosine_similarities = cosine_similarity(title_embedding.reshape(1, -1), topics_embeddings).flatten()

    # Поиск индекса темы с наибольшим значением косинусного сходства
    closest_topic_index = np.argmax(cosine_similarities)

    # Возвращение ближайшей темы и значения косинусного сходства
    return topics[closest_topic_index], float(cosine_similarities[closest_topic_index])

@app.post("/find-closest-topic")
def find_closest_topic_endpoint(request_body: RequestBody):
    title = request_body.title
    topics = request_body.topics

    if not topics:
        raise HTTPException(status_code=400, detail="The topics list cannot be empty.")

    closest_topic, accuracy = find_closest_topic(title, topics)

    return {
        "closest_topic": closest_topic,
        "accuracy": accuracy
    }

# Для локального запуска сервера, добавьте этот код в конце файла:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
