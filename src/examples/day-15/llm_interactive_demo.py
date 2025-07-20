"""
Day 15 Advanced Example: LLM Interactive Demo
This script lets you interact with a decoder-only LLM (GPT-2) and visualizes next-token probabilities.
"""
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt

# 1. Interactive text generation
generator = pipeline('text-generation', model='gpt2')
print("\nEnter a prompt for GPT-2 (or press Enter to use a default prompt):")
prompt = input().strip()
if not prompt:
    prompt = "The future of artificial intelligence is"
print(f"\nGPT-2 completion:\n{generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']}")

# 2. Visualize next-token probabilities for a sample prompt
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
sample_prompt = "The future of artificial intelligence is"
inputs = tokenizer(sample_prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=0)

# Get top 10 next tokens
topk = 10
probs_topk, indices_topk = torch.topk(probs, topk)
tokens_topk = [tokenizer.decode([idx]) for idx in indices_topk]

plt.figure(figsize=(10,5))
plt.bar(tokens_topk, probs_topk.numpy())
plt.title(f"GPT-2 Next Token Probabilities\nPrompt: '{sample_prompt}'")
plt.xlabel('Next Token')
plt.ylabel('Probability')
plt.tight_layout()
plt.show() 