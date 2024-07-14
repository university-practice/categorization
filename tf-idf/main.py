from fastapi import FastAPI

app = FastAPI()

@app.post("/tf-idf")
async def read_root(request: dict):

    print(request)
    for i in request["topics"]:
        print(i)

    return "hello1"



