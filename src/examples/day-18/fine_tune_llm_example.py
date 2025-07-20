"""
Day 18 Example: Fine-tuning a Large Language Model (LLM) for Sentiment Classification

This script demonstrates the conceptual workflow for fine-tuning a pre-trained LLM (e.g., BERT)
on a small sentiment classification task using Hugging Face Transformers.

Note: This is a minimal, educational example. For real fine-tuning, use a larger dataset and proper hardware.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate

# 1. Data Preparation (Dummy dataset)
dummy_data = {
    "text": [
        "This product is amazing!",
        "I am very disappointed with the service.",
        "It works perfectly, highly recommend.",
        "Never buying from here again.",
        "Neutral opinion, nothing special."
    ],
    "label": [1, 0, 1, 0, 0]  # 1 = positive, 0 = negative/neutral
}
raw_dataset = Dataset.from_dict(dummy_data)

# 2. Model and Tokenizer Selection
model_name = "distilbert-base-uncased"  # Small, fast model for demo
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 3. Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

# 4. Train/test split
split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
test_dataset = split["test"]

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=5,
    report_to=[],  # Disable logging to external services
)

# 6. Metrics function
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning (this is a mock run, training is commented out)...")
# Uncomment the next line to actually train (requires GPU for speed):
# trainer.train()
print("Fine-tuning complete (mock run).\n")

# 8. Evaluation (mock)
# results = trainer.evaluate()
# print(f"\nEvaluation results: {results}")

# 9. Inference example (mock)
# from transformers import pipeline
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# print(classifier("This is an excellent movie!"))

print("Example complete. See comments for real training and inference steps.") 