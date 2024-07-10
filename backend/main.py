from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
   return {"message": "Hello, World!"}

@app.post("//api/categorize")
def read_root():
	# read body

	# read file

	# http request bert:80801/predict
	# response1 0.8

	# http request tf-idf:80801/predict
	# response2 0.7

	# merge results

	# work with file

	# return file
    return {"message": "Hello, World!"}
