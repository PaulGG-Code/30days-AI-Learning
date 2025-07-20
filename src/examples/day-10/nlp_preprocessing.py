import re
import string
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import os

print(f"NLTK version: {nltk.__version__}")
# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

class NLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def to_lowercase(self, tokens):
        """Convert tokens to lowercase"""
        return [token.lower() for token in tokens]
    
    def remove_punctuation(self, tokens):
        """Remove punctuation from tokens"""
        return [token for token in tokens if token not in string.punctuation]
    
    def remove_stopwords(self, tokens):
        """Remove stop words from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def pos_tag_tokens(self, tokens):
        """Apply POS tagging to tokens"""
        return pos_tag(tokens)
    
    def preprocess_text(self, text, steps=None):
        """Complete preprocessing pipeline"""
        if steps is None:
            steps = ['tokenize', 'lowercase', 'remove_punct', 'remove_stopwords']
        
        result = text
        print(f"Original text: {text}")
        
        if 'tokenize' in steps:
            result = self.tokenize(result)
            print(f"After tokenization: {result}")
        
        if 'lowercase' in steps:
            result = self.to_lowercase(result)
            print(f"After lowercasing: {result}")
        
        if 'remove_punct' in steps:
            result = self.remove_punctuation(result)
            print(f"After removing punctuation: {result}")
        
        if 'remove_stopwords' in steps:
            result = self.remove_stopwords(result)
            print(f"After removing stop words: {result}")
        
        if 'stem' in steps:
            result = self.stem_tokens(result)
            print(f"After stemming: {result}")
        
        if 'lemmatize' in steps:
            result = self.lemmatize_tokens(result)
            print(f"After lemmatization: {result}")
        
        if 'pos_tag' in steps:
            result = self.pos_tag_tokens(result)
            print(f"After POS tagging: {result}")
        
        return result

def analyze_text_statistics(text):
    """Analyze basic text statistics"""
    # Tokenize into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Remove punctuation for word analysis
    words_no_punct = [word for word in words if word not in string.punctuation]
    
    # Calculate statistics
    stats = {
        'total_characters': len(text),
        'total_sentences': len(sentences),
        'total_words': len(words_no_punct),
        'unique_words': len(set(word.lower() for word in words_no_punct)),
        'avg_sentence_length': len(words_no_punct) / len(sentences) if sentences else 0,
        'avg_word_length': sum(len(word) for word in words_no_punct) / len(words_no_punct) if words_no_punct else 0
    }
    
    return stats, words_no_punct

def plot_word_frequency(words, top_n=10):
    """Plot word frequency distribution"""
    word_freq = Counter(word.lower() for word in words)
    most_common = word_freq.most_common(top_n)
    
    words_list, frequencies = zip(*most_common)
    
    plt.figure(figsize=(12, 6))
    plt.bar(words_list, frequencies)
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs('ai-llm-agent-course/examples/day-10', exist_ok=True)
    plt.savefig('ai-llm-agent-course/examples/day-10/word_frequency.png', dpi=150)
    plt.show()
    
    return most_common

if __name__ == "__main__":
    # Sample text for demonstration
    sample_text = """
    Natural Language Processing (NLP) is a fascinating field of artificial intelligence 
    that focuses on the interaction between computers and humans through natural language. 
    The ultimate objective of NLP is to read, decipher, understand, and make sense of 
    human languages in a manner that is valuable. NLP combines computational linguistics 
    with statistical, machine learning, and deep learning models to enable computers to 
    process human language in the form of text or voice data.
    """
    
    print("=== NLP Preprocessing Example ===\n")
    
    # Initialize preprocessor
    preprocessor = NLPPreprocessor()
    
    # Demonstrate step-by-step preprocessing
    print("1. Step-by-step preprocessing:")
    print("-" * 50)
    processed_tokens = preprocessor.preprocess_text(
        sample_text.strip(), 
        steps=['tokenize', 'lowercase', 'remove_punct', 'remove_stopwords']
    )
    
    print("\n2. Comparing stemming vs lemmatization:")
    print("-" * 50)
    sample_words = ['running', 'ran', 'runs', 'easily', 'fairly', 'better', 'good']
    print(f"Original words: {sample_words}")
    
    stemmed = [preprocessor.stemmer.stem(word) for word in sample_words]
    lemmatized = [preprocessor.lemmatizer.lemmatize(word) for word in sample_words]
    
    print(f"Stemmed: {stemmed}")
    print(f"Lemmatized: {lemmatized}")
    
    print("\n3. POS Tagging example:")
    print("-" * 50)
    sample_sentence = "The quick brown fox jumps over the lazy dog."
    tokens = word_tokenize(sample_sentence)
    pos_tags = pos_tag(tokens)
    print(f"Sentence: {sample_sentence}")
    print(f"POS Tags: {pos_tags}")
    
    print("\n4. Text statistics:")
    print("-" * 50)
    stats, words = analyze_text_statistics(sample_text)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n5. Word frequency analysis:")
    print("-" * 50)
    most_common = plot_word_frequency(words, top_n=10)
    print("Most frequent words:")
    for word, freq in most_common:
        print(f"  {word}: {freq}")
    
    print("\n6. Complete preprocessing pipeline:")
    print("-" * 50)
    final_result = preprocessor.preprocess_text(
        "The researchers are studying machine learning algorithms!",
        steps=['tokenize', 'lowercase', 'remove_punct', 'remove_stopwords', 'lemmatize']
    )
    print(f"Final processed tokens: {final_result}")

