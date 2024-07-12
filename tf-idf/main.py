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
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([title] + topics)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    closest_topic_index = np.argmax(cosine_similarities)
    return topics[closest_topic_index]

@app.post("/find-closest-topic")
def find_closest_topic_endpoint(request_body: RequestBody):
    title = request_body.title
    topics = request_body.topics

    if not topics:
        raise HTTPException(status_code=400, detail="The topics list cannot be empty.")

    closest_topic = find_closest_topic(title, topics)
    return {"closest_topic": closest_topic}

# Запуск сервера командой (если запускаете локально, добавьте этот код в конце файла):
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
