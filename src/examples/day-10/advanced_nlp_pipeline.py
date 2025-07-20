"""
Day 10 Advanced Example: End-to-End NLP Pipeline for Sentiment Analysis
This script demonstrates text preprocessing, feature extraction, and sentiment classification using NLTK and scikit-learn.
"""
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample dataset (text, label)
data = [
    ("I love this product! It's amazing and works great.", 1),
    ("Absolutely terrible, waste of money.", 0),
    ("Best purchase I've made this year.", 1),
    ("I hate it. Completely useless.", 0),
    ("Not bad, but could be better.", 0),
    ("Fantastic! Exceeded my expectations.", 1),
    ("Awful experience, will not buy again.", 0),
    ("Very happy with the quality.", 1),
    ("Disappointing, broke after a week.", 0),
    ("Superb! Highly recommend it.", 1),
]
texts, labels = zip(*data)

# Preprocessing function
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Preprocess all texts
texts_clean = [preprocess(t) for t in texts]

# Feature extraction: TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts_clean)
y = labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Word clouds for each sentiment
pos_text = ' '.join([texts_clean[i] for i in range(len(y)) if y[i] == 1])
neg_text = ' '.join([texts_clean[i] for i in range(len(y)) if y[i] == 0])
wordcloud_pos = WordCloud(width=400, height=200, background_color='white').generate(pos_text)
wordcloud_neg = WordCloud(width=400, height=200, background_color='white').generate(neg_text)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')
plt.subplot(1,2,2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')
plt.tight_layout()
plt.show() 