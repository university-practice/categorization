from fastapi import FastAPI

app = FastAPI()


@app.post("/bert")
async def read_root(request: dict):

    print(request)

    return "hello"



