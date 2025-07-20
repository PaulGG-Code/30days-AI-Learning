"""
Day 19 Example: Applications of LLMs - Beyond Chatbots

This script demonstrates several practical applications of Large Language Models (LLMs)
using Hugging Face Transformers pipelines. Each section shows a different real-world use case.

Note: For demo purposes, small models are used. Requires 'transformers' and 'torch' installed.
"""

from transformers import pipeline

# 1. Content Generation (Text Generation)
print("\n--- Content Generation ---")
generator = pipeline("text-generation", model="distilgpt2")
prompt = "Once upon a time in a world of AI,"
result = generator(prompt, max_length=40, num_return_sequences=1)
print(f"Prompt: {prompt}\nGenerated: {result[0]['generated_text']}")

# 2. Question Answering
print("\n--- Question Answering ---")
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human language."
question = "What are LLMs?"
answer = qa(question=question, context=context)
print(f"Q: {question}\nA: {answer['answer']}")

# 3. Code Explanation (Text2Text Generation)
print("\n--- Code Explanation ---")
explainer = pipeline("text2text-generation", model="google/flan-t5-small")
code_snippet = "def add(a, b):\n    return a + b"
prompt = f"Explain this Python code: {code_snippet}"
explanation = explainer(prompt, max_length=50)[0]['generated_text']
print(f"Code: {code_snippet}\nExplanation: {explanation}")

# 4. Sentiment Analysis
print("\n--- Sentiment Analysis ---")
sentiment = pipeline("sentiment-analysis")
text = "I absolutely love this product!"
result = sentiment(text)[0]
print(f"Text: {text}\nSentiment: {result['label']} (score: {result['score']:.2f})")

# 5. Text Simplification (Paraphrasing)
print("\n--- Text Simplification ---")
paraphraser = pipeline("text2text-generation", model="google/flan-t5-small")
complex_text = "The precipitation will be intermittent throughout the duration of the afternoon."
simple_prompt = f"Simplify: {complex_text}"
simplified = paraphraser(simple_prompt, max_length=50)[0]['generated_text']
print(f"Original: {complex_text}\nSimplified: {simplified}")

print("\nDemo complete. Each section shows a different LLM application.") 