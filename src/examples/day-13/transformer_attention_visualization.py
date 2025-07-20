"""
Day 13 Advanced Example: Visualizing Transformer Self-Attention
This script uses Hugging Face Transformers to classify text and visualizes self-attention weights for a sample sentence.
"""
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

# Sample sentence
sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors='pt')

# Forward pass with attention
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple: (num_layers, batch, num_heads, seq_len, seq_len)

# Visualize attention from the last layer, first head
attn = attentions[-1][0,0].numpy()  # shape: (seq_len, seq_len)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

plt.figure(figsize=(8,7))
plt.imshow(attn, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.title('BERT Self-Attention (Last Layer, Head 0)')
plt.xlabel('Key (Attended To)')
plt.ylabel('Query (Paying Attention)')
plt.tight_layout()
plt.show()

# For classification, use pipeline (optional)
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print('Sentiment:', classifier(sentence)[0]) 