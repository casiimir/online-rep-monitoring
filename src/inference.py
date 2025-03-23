from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.special import softmax

# Model name (RoBERTa fine-tuned sentiment analysis)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def preprocess(text: str) -> str:
    """
    Text processing.
    """
    processed_tokens = []
    for token in text.split():
        if token.startswith('@') and len(token) > 1:
            processed_tokens.append('@user')
        elif token.startswith('http'):
            processed_tokens.append('http')
        else:
            processed_tokens.append(token)
    return " ".join(processed_tokens)

def predict_sentiment(text: str) -> dict:
    """
    Return a dict. with sentiment and scores.
    """
    # Text preprocessing
    preprocessed_text = preprocess(text)
    
    # Input tokenization
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
    
    # Run inference
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Points extraction and softmax
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    
    # Mapping id->label (0: negative, 1: neutral, 2: positive)
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    
    ranking = np.argsort(scores)[::-1]
    
    results = {id2label[i]: round(float(scores[i]), 4) for i in ranking}
    return results

if __name__ == "__main__":
    example_text = "Che spettacolo, funziona!"
    sentiment = predict_sentiment(example_text)
    print(f"Input: {example_text}")
    print("Sentiment:", sentiment)
