"""
Day 9 Advanced Example: LSTM for Character-Level Text Generation
This script builds and trains an LSTM to generate text character-by-character using TensorFlow/Keras.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from collections import Counter

# Sample text corpus (can be replaced with any text)
text = (
    "In the beginning, AI was just a dream. Now, it is a reality. "
    "The explorer journeys deeper into the world of artificial intelligence. "
    "Neural networks, deep learning, and language models await. "
)

# Create character-to-index and index-to-character mappings
chars = sorted(list(set(text)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

# Prepare input sequences and targets
seq_length = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

X = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.float32)
y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char2idx[char]] = 1
    y[i, char2idx[next_chars[i]]] = 1

# Build the LSTM model
model = models.Sequential([
    layers.LSTM(128, input_shape=(seq_length, len(chars))),
    layers.Dense(len(chars), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
history = model.fit(X, y, batch_size=64, epochs=20)

# Plot training loss curve
plt.figure(figsize=(7,4))
plt.plot(history.history['loss'])
plt.title('Training Loss (LSTM Text Generation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Text generation function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed, length=200, temperature=0.5):
    generated = seed
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed):
            if char in char2idx:
                x_pred[0, t, char2idx[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx2char[next_index]
        generated += next_char
        seed = seed[1:] + next_char
    return generated

# Generate sample text
seed_text = text[:seq_length]
generated_text = generate_text(seed_text, length=300, temperature=0.5)
print("\n--- Generated Text ---")
print(generated_text)

# Plot histogram of generated character frequencies
char_counts = Counter(generated_text)
plt.figure(figsize=(10,4))
plt.bar(char_counts.keys(), char_counts.values())
plt.title('Character Frequency in Generated Text')
plt.xlabel('Character')
plt.ylabel('Count')
plt.tight_layout()
plt.show() 