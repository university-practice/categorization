from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

class RequestBody(BaseModel):
    title: str
    topics: list

def find_closest_topic(title, topics):
    # Создание модели TF-IDF и вычисление матрицы TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([title] + topics)

    # Вычисление косинусного сходства между заголовком и всеми темами
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Поиск индекса темы с наибольшим значением косинусного сходства
    closest_topic_index = np.argmax(cosine_similarities)

    # Возвращение ближайшей темы и значения косинусного сходства
    return topics[closest_topic_index], cosine_similarities[closest_topic_index]

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

# Запуск сервера командой (если запускаете локально, добавьте этот код в конце файла):
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
