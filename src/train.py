import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load sentiment140 dataset (using the "train" split)
dataset = load_dataset("sentiment140", split="train", verification_mode="no_checks")

# Remap labels: sentiment140 labels are 0 (negative), 2 (neutral), 4 (positive)
def remap_labels(example):
    mapping = {0: 0, 2: 1, 4: 2}
    example["sentiment"] = mapping[example["sentiment"]]
    return example

dataset = dataset.map(remap_labels)
# Rename the "sentiment" column to "label" for consistency
dataset = dataset.rename_column("sentiment", "label")

# Split the dataset into train/validation/test (80/10/10)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split_dataset["train"].train_test_split(test_size=0.125, seed=42)  # 0.125 of 80% equals 10% of total
train_dataset = train_val["train"]
validation_dataset = train_val["test"]
test_dataset = split_dataset["test"]

# Use a subset of the datasets for faster training (debugging)
train_dataset = train_dataset.select(range(5000))
validation_dataset = validation_dataset.select(range(1000))
test_dataset = test_dataset.select(range(1000))

print("Train dataset:", train_dataset)
print("Validation dataset:", validation_dataset)
print("Test dataset:", test_dataset)

# Load the tokenizer and model, setting num_labels to 3
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Preprocessing function: tokenize the tweet text
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply the preprocessing in batched mode
train_dataset = train_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set the dataset format for PyTorch (only keep necessary columns)
columns = ["input_ids", "attention_mask", "label"]
train_dataset.set_format(type="torch", columns=columns)
validation_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

# Configure TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",             # Directory to save results and checkpoints
    evaluation_strategy="epoch",          # Evaluate at the end of each epoch
    save_strategy="epoch",                # Save checkpoint at the end of each epoch
    num_train_epochs=3,                   # Number of training epochs (set to 1 for debugging)
    per_device_train_batch_size=64,      # Training batch size per device
    per_device_eval_batch_size=32,        # Evaluation batch size per device
    logging_dir="./logs",                 # Directory for logs
    load_best_model_at_end=True,          # Load the best model at the end of training
    metric_for_best_model="accuracy",     # Metric used to compare models
)

# Function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
