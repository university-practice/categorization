from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import requests
import pandas as pd
from io import BytesIO, StringIO

from starlette.responses import StreamingResponse

app = FastAPI()


@app.post("/api/categorize")
async def read_root(
	columnName: str = Form(),
	topics: List[str] = Form(...),
	xlsxFile: UploadFile = File(...)
):
    print("topics", topics)
    try:
        file_content = await xlsxFile.read()
        df = pd.read_excel(BytesIO(file_content))
        output = BytesIO()

        data = df.dropna(subset=[columnName])
        print("TEST", data['Направление работы конференции'].unique().tolist())
        for index, row in data.iterrows():
            response1_data = None
            response2_data = None

            try:
                response1 = requests.get("https://jsonplaceholder.typicode.com/posts/1")
                response1.raise_for_status()
                response1_data = response1.json()
            except requests.exceptions.RequestException as e:
                print(f"Error with first fake request: {e}")

            try:
                response2 = requests.get("https://jsonplaceholder.typicode.com/posts/2")
                response2.raise_for_status()
                response2_data = response2.json()
            except requests.exceptions.RequestException as e:
                print(f"Error with second fake request: {e}")

            data.at[index, 'prediction'] = row[columnName]

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
