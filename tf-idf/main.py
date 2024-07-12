from fastapi import FastAPI

app = FastAPI()

@app.post("/")
async def read_root():
    return "hello"
