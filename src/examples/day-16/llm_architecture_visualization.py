"""
Day 16 Advanced Example: LLM Architecture Visualization
This script compares BERT (encoder-only), GPT-2 (decoder-only), and T5 (encoder-decoder) on a sample input, and visualizes their outputs and attention maps.
"""
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np

# 1. BERT: Encoder-only (contextual embeddings)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
sample_text = "Artificial intelligence is transforming the world."
inputs_bert = bert_tokenizer(sample_text, return_tensors='pt')
with torch.no_grad():
    outputs_bert = bert_model(**inputs_bert)
    last_hidden = outputs_bert.last_hidden_state[0]  # (seq_len, hidden_dim)
    attentions_bert = outputs_bert.attentions[-1][0].mean(0).numpy()  # (seq_len, seq_len)
tokens_bert = bert_tokenizer.convert_ids_to_tokens(inputs_bert['input_ids'][0])

plt.figure(figsize=(8,7))
plt.imshow(attentions_bert, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(tokens_bert)), tokens_bert, rotation=90)
plt.yticks(range(len(tokens_bert)), tokens_bert)
plt.title('BERT (Encoder-only) Self-Attention (Last Layer, Mean Heads)')
plt.tight_layout()
plt.show()

# 2. GPT-2: Decoder-only (text generation)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
inputs_gpt2 = gpt2_tokenizer(sample_text, return_tensors='pt')
with torch.no_grad():
    outputs_gpt2 = gpt2_model(**inputs_gpt2)
    attentions_gpt2 = outputs_gpt2.attentions[-1][0,0].numpy()  # (seq_len, seq_len)
    generated = gpt2_model.generate(inputs_gpt2['input_ids'], max_length=20)
generated_text = gpt2_tokenizer.decode(generated[0])
tokens_gpt2 = gpt2_tokenizer.convert_ids_to_tokens(inputs_gpt2['input_ids'][0])

plt.figure(figsize=(8,7))
plt.imshow(attentions_gpt2, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(tokens_gpt2)), tokens_gpt2, rotation=90)
plt.yticks(range(len(tokens_gpt2)), tokens_gpt2)
plt.title('GPT-2 (Decoder-only) Self-Attention (Last Layer, Head 0)')
plt.tight_layout()
plt.show()

print(f"\nGPT-2 generated text:\n{generated_text}")

# 3. T5: Encoder-Decoder (text transformation)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
input_text = "summarize: Artificial intelligence is transforming the world by enabling new capabilities."
inputs_t5 = t5_tokenizer(input_text, return_tensors='pt')
with torch.no_grad():
    summary_ids = t5_model.generate(inputs_t5['input_ids'], max_length=20)
summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"\nT5 summary:\n{summary}")

# Note: T5 does not expose attention weights in generate() by default, so we skip T5 attention visualization here.
print("\n[Note] T5 attention visualization is not shown because generate() does not return attention weights by default. Advanced users can use output_attentions=True and custom decoding for research purposes.") 