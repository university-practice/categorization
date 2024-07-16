from typing import TypedDict, List, DefaultDict, Dict
from collections import defaultdict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from io import BytesIO, StringIO
from topics import topics_v1

from starlette.responses import StreamingResponse

load_dotenv()

app = FastAPI()

class Prediction(TypedDict):
    predict: str
    accuracy: float
    weight: float

def select_prediction(preds: List[Prediction]) -> Dict[str, float]:
    scores: DefaultDict[str, float] = defaultdict(float)
    max_accuracy: float = 0.0

    # Суммируем оценки предсказаний
    for pred in preds:
        prediction: str = pred['predict']
        score: float = pred['accuracy'] * pred['weight']
        scores[prediction] += score
        if pred['accuracy'] > max_accuracy:
            max_accuracy = pred['accuracy']

    closest_topic: str = max(scores, key=scores.get)

    max_probability: float = scores[closest_topic]

    if max_probability > 1:
        max_probability /= sum(scores.values())
    else:
        max_probability = min(max_probability, 1.0)

    result = {
        "closest_topic": closest_topic,
        "accuracy": max_probability
    }

    return result

@app.post("/api/categorize")
async def read_root(
	columnName: str = Form(),
	topics: List[str] = Form(...),
	xlsxFile: UploadFile = File(...),
    mode: str = Form(), # bert | any string
    threshold: float = Form()
):
    try:
        file_content = await xlsxFile.read()
        df = pd.read_excel(BytesIO(file_content))
        output = BytesIO()

        data = df.dropna(subset=[columnName])
        for index, row in data.iterrows():
            bert_data = None
            tf_idf_data = None
            embedding_data = None



            tf_idf_url = os.getenv("TF_IDF_URL")
            if not tf_idf_url:
                return {"error": "TF_IDF not set in env"}

            embedding_url = os.getenv("EMBEDDING_URL")
            if not embedding_url:
                return {"error": "EMBEDDING_URL not set in env"}

            request = {
                "title": row[columnName],
                "topics": topics_v1 if mode == "bert" else topics
            }

            # TF-IDF
            try:
                tf_idf_response = requests.post(tf_idf_url, json=request)
                tf_idf_response.raise_for_status()
                tf_idf_data = tf_idf_response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error with second fake request: {e}")

            # EMBEDDING
            try:
                embedding_response = requests.post(embedding_url, json=request)
                embedding_response.raise_for_status()
                embedding_data = embedding_response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error with second fake request: {e}")

			# BERT
            if (mode == "bert"):
                bert_url = os.getenv("BERT_URL")
                if not bert_url:
                    return {"error": "BERT URL not set in env"}
                request = {
                    "text": row[columnName]
                }
                try:
                    bert_response = requests.post(bert_url, json=request)
                    bert_response.raise_for_status()
                    bert_data = bert_response.json()
                    print("FROM BERT:", bert_data)
                except requests.exceptions.RequestException as e:
                    print(f"Error with second fake request: {e}")

            if tf_idf_data is not None and embedding_data is not None:
                best_prediction = select_prediction([
	                {
	                    "predict": tf_idf_data["closest_topic"],
	                    "accuracy": tf_idf_data["accuracy"],
	                    "weight": 0.5
	                },
					{
	                    "predict": embedding_data["closest_topic"],
	                    "accuracy": embedding_data["accuracy"],
	                    "weight": 1
	                }
	            ])
                data.at[index, 'prediction'] = best_prediction["closest_topic"] if best_prediction["accuracy"] >= threshold else ""
            else:
                data.at[index, 'prediction'] = 'Unknown'

        # Запишите DataFrame обратно в BytesIO буфер
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='Sheet1')

        output.seek(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing xlsx file") from e
    finally:
        xlsxFile.file.close()

    return StreamingResponse(
        output,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": "attachment; filename=processed_file.xlsx"}
    )
