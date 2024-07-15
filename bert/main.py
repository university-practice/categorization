from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig


app = FastAPI()

class TextInput(BaseModel):
    text: str

# Загрузка модели и токенайзера
model_path = "./model/model_v0_1.pt"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Создание конфигурации модели
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

# Создание экземпляра модели
model = DistilBertModel(config)

# Загрузка весов модели с указанием map_location
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


@app.post("/predict")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt")
    outputs = model(**inputs)

    return outputs
