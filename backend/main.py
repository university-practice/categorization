from fastapi import FastAPI, UploadFile, File, Form
import requests
import pandas as pd
from io import StringIO

app = FastAPI()


@app.post("/api/categorize")
async def read_root(columnName: str = Form(), csvFile: UploadFile = File(...)):

	# http request bert:80801/predict
	# response1 0.8

    response = requests.get('https://jsonplaceholder.typicode.com/todos/1')

    if response.status_code == 200:
        print(response.content)
    elif response.status_code == 404:
        print('Not Found.')

	# http request tf-idf:80801/predict
	# response2 0.7

	# merge results

	# work with file
    df = pd.read_csv(StringIO((await csvFile.read()).decode()))
    csvFile.file.close()
    df['prediction'] = 'prediction'

    # return file
    return df.to_csv(index=False)
