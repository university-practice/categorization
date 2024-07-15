from typing import List
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

@app.post("/api/categorize")
async def read_root(
	columnName: str = Form(),
	topics: List[str] = Form(...),
	xlsxFile: UploadFile = File(...),
    mode: str = Form() # bert | any string
):
    try:
        file_content = await xlsxFile.read()
        df = pd.read_excel(BytesIO(file_content))
        output = BytesIO()

        data = df.dropna(subset=[columnName])
        for index, row in data.iterrows():
            response1_data = None
            tf_idf_data = None

            if (mode == "bert"):
                print("BERT API")
	            # try:
	            #     response1 = requests.get("https://jsonplaceholder.typicode.com/posts/1")
	            #     response1.raise_for_status()
	            #     response1_data = response1.json()
	            # except requests.exceptions.RequestException as e:
	            #     print(f"Error with first fake request: {e}")

            url = os.getenv("TF_IDF_URL")
            if not url:
                return {"error": "TF_IDF not set in env"}

            request = {
                "title": row[columnName],
                "topics": topics_v1 if mode == "bert" else topics
            }
            try:
                tf_idf_response = requests.post(url, json=request)
                tf_idf_response.raise_for_status()
                tf_idf_data = tf_idf_response.json()
                print(tf_idf_data)
            except requests.exceptions.RequestException as e:
                print(f"Error with second fake request: {e}")

            if tf_idf_data is not None:
                data.at[index, 'prediction'] = tf_idf_data["closest_topic"]
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
