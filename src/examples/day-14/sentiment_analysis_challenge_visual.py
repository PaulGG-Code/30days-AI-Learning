"""
Day 14 Advanced Example: Sentiment Analysis Challenge with Visualizations
This script builds a sentiment analysis model with pre-trained embeddings and visualizes training, embeddings, and results.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import string
from nltk.corpus import stopwords
from sklearn.decomposition import PCA

# --- 1. Prepare Data (Small Custom Dataset for Demonstration) ---
texts = [
    "This movie was fantastic! I loved every minute of it.",
    "Absolutely brilliant and captivating. A must-watch.",
    "The acting was superb, but the plot was a bit slow.",
    "Not bad, but I expected more. It was just okay.",
    "Terrible film. A complete waste of time and money.",
    "I hated it. The worst movie I've seen this year."
]
labels = np.array([1, 1, 0, 0, 0, 0]) # 1 for positive, 0 for negative/neutral

# --- 2. Text Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

import nltk
nltk.download('stopwords')
preprocessed_texts = [preprocess_text(text) for text in texts]

# --- 3. Simulate Pre-trained Word Embeddings ---
all_words = set()
for text in preprocessed_texts:
    for word in text.split():
        all_words.add(word)
embedding_dim = 50
word_embedding_map = {word: np.random.rand(embedding_dim) for word in all_words}

# --- 4. Tokenize and Create Embedding Matrix ---
tokenizer = Tokenizer(num_words=len(all_words) + 1)
tokenizer.fit_on_texts(preprocessed_texts)
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = word_embedding_map.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# --- 5. Convert Text to Sequences ---
sequences = tokenizer.texts_to_sequences(preprocessed_texts)
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.3, random_state=42)

# --- 6. Build and Train Model ---
model = Sequential([
    Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_test, y_test))

# --- 7. Visualize Training Curves ---
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

# --- 8. Visualize Embedding Space (PCA) ---
embeddings = embedding_matrix[1:len(all_words)+1]  # skip padding idx 0
words = list(word_index.keys())
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
plt.figure(figsize=(8,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1], c='blue')
for i, word in enumerate(words):
    plt.text(emb_2d[i,0]+0.01, emb_2d[i,1]+0.01, word, fontsize=12)
plt.title('Word Embeddings Visualized (PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 9. Evaluate and Show Confusion Matrix ---
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show() 