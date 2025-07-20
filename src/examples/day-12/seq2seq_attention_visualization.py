"""
Day 12 Advanced Example: Seq2Seq with Attention Visualization
This script builds a simple Seq2Seq model with attention (Keras) on a toy sequence reversal task and visualizes attention weights.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Toy data: sequence reversal (e.g., input: [1,2,3,4], output: [4,3,2,1])
num_samples = 10000
seq_length = 7
vocab_size = 10  # digits 0-9
X = np.random.randint(1, vocab_size, size=(num_samples, seq_length))
y = np.flip(X, axis=1)

# Split
X_train, X_test = X[:8000], X[8000:]
y_train, y_test = y[:8000], y[8000:]

# Model
embedding_dim = 32
units = 64

# Encoder
encoder_inputs = layers.Input(shape=(seq_length,))
encoder_emb = layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = layers.LSTM(units, return_sequences=True, return_state=True)(encoder_emb)

# Decoder
decoder_inputs = layers.Input(shape=(seq_length,))
decoder_emb = layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=[state_h, state_c])

# Attention
attention = layers.AdditiveAttention()
attention_out = attention([decoder_outputs, encoder_outputs])
concat = layers.Concatenate()([decoder_outputs, attention_out])
dense = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(concat)

model = models.Model([encoder_inputs, decoder_inputs], dense)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare decoder input/output (teacher forcing)
decoder_input_train = np.zeros_like(y_train)
decoder_input_train[:,1:] = y_train[:,:-1]
# Start token is 0
decoder_input_test = np.zeros_like(y_test)
decoder_input_test[:,1:] = y_test[:,:-1]

# Train
history = model.fit([X_train, decoder_input_train], np.expand_dims(y_train, -1), epochs=10, batch_size=128, validation_split=0.1)

# Plot accuracy and loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize attention weights for a sample
# Build a sub-model to get attention scores
encoder_output_layer = model.get_layer(index=3)  # encoder_outputs
decoder_output_layer = model.get_layer(index=6)  # decoder_outputs
attention_layer = model.get_layer(index=7)       # AdditiveAttention

# Build a new model to output attention scores
attention_model = models.Model(
    inputs=model.inputs,
    outputs=attention_layer([decoder_output_layer.output, encoder_output_layer.output])
)

sample_idx = 0
sample_in = X_test[sample_idx:sample_idx+1]
sample_dec = decoder_input_test[sample_idx:sample_idx+1]
attn_scores = attention_model.predict([sample_in, sample_dec])[0]

plt.figure(figsize=(7,6))
plt.imshow(attn_scores, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Attention Weights Heatmap (Sample)')
plt.xlabel('Encoder Time Step')
plt.ylabel('Decoder Time Step')
plt.tight_layout()
plt.show()

print('Input sequence:', X_test[sample_idx])
print('Target (reversed):', y_test[sample_idx])
print('Model prediction:', np.argmax(model.predict([sample_in, sample_dec])[0], axis=-1)[0]) 