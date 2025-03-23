from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained("./model_checkpoint")
model = AutoModelForSequenceClassification.from_pretrained("./model_checkpoint")
model.eval()

# Define Prometheus metrics
REQUEST_COUNTER = Counter("inference_requests_total", "Total number of inference requests")
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "Inference request latency in seconds")

# Define request body schema
class PredictRequest(BaseModel):
    text: str

# Define /predict endpoint for inference
@app.post("/predict")
def predict(request: PredictRequest):
    REQUEST_COUNTER.inc()
    with REQUEST_LATENCY.time():
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].detach().numpy()
        # Compute softmax scores
        exp_logits = np.exp(logits)
        scores = exp_logits / np.sum(exp_logits)
        # Map label to sentiment (0: negative, 1: neutral, 2: positive)
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
        pred_label = int(np.argmax(scores))
        result = {"label": mapping[pred_label], "scores": scores.tolist()}
    return result

if __name__ == "__main__":
    import uvicorn
    # Start a Prometheus metrics server on port 8001
    start_http_server(8001)
    # Run the FastAPI app on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
