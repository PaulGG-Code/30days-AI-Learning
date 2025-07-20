"""
Day 25 Example: Train and Save a Simple Text Classifier (AG News)
This script downloads the AG News dataset, trains a simple Keras model, saves it, loads it, and tests it.
"""
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os

# 1. Load AG News dataset
print("Loading AG News dataset from Hugging Face...")
dataset = load_dataset('ag_news')
train_texts = [x['text'] for x in dataset['train']]
train_labels = [x['label'] for x in dataset['train']]
test_texts = [x['text'] for x in dataset['test']]
test_labels = [x['label'] for x in dataset['test']]

# For speed, use a subset (e.g., 5000 train, 1000 test)
N_train, N_test = 5000, 1000
train_texts, train_labels = train_texts[:N_train], train_labels[:N_train]
test_texts, test_labels = test_texts[:N_test], test_labels[:N_test]

# 2. Preprocess: Tokenize and pad
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)
X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
y_train = to_categorical(train_labels, num_classes=4)
y_test = to_categorical(test_labels, num_classes=4)

# 3. Build the model
model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# 4. Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 5. Visualize training
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Save the model
os.makedirs('examples/day-25/model', exist_ok=True)
model.save('examples/day-25/model/agnews_text_classifier.h5')
print("Model saved to examples/day-25/model/agnews_text_classifier.h5")

# 7. Load the model and test
loaded_model = load_model('examples/day-25/model/agnews_text_classifier.h5')
y_pred_proba = loaded_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=['World', 'Sports', 'Business', 'Sci/Tech']))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['World', 'Sports', 'Business', 'Sci/Tech'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# 8. Test on new data
sample_texts = [
    "NASA launches a new satellite to study climate change.",
    "The stock market saw a significant increase today.",
    "The World Cup final was an exciting match.",
    "Apple releases its latest smartphone with new features."
]
sample_seq = tokenizer.texts_to_sequences(sample_texts)
sample_seq = pad_sequences(sample_seq, maxlen=max_len, padding='post', truncating='post')
sample_pred = loaded_model.predict(sample_seq)
sample_pred_labels = np.argmax(sample_pred, axis=1)
label_map = ['World', 'Sports', 'Business', 'Sci/Tech']
print("\nSample predictions:")
for text, label_idx in zip(sample_texts, sample_pred_labels):
    print(f"Text: {text}\nPredicted label: {label_map[label_idx]}\n") 