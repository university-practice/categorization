from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.act = nn.ReLU(0.2)
        self.drop = nn.Dropout(0.12)
        self.linear_1 = nn.Linear(self.bert.config.hidden_size * 2, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 21)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        last_hidden_state = self.bert(input_ids=input_id, attention_mask=mask).last_hidden_state
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        logits = self.linear_1(mean_max_embeddings)
        logits = self.drop(self.act(logits))
        logits = self.linear_2(logits)
        logits = self.act(logits)
        logits = self.linear_3(logits)
        final_layer = self.softmax(logits)
        return final_layer

class TextInput(BaseModel):
    text: str

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = BertClassifier("cointegrated/rubert-tiny2")

model_path = "./model/model_v0_1.pt"
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model state dict loaded successfully.")
except Exception as e:
    print(f"Error loading model state dict: {e}")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@app.post("/predict")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt")
    input_ids = inputs['input_ids'].squeeze(1)
    attention_mask = inputs['attention_mask'].squeeze(1)

    with torch.no_grad():
        output = model(input_id=input_ids, mask=attention_mask)

    return output.detach().numpy().tolist()
