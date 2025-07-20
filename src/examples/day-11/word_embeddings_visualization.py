"""
Day 11 Advanced Example: Visualizing Word Embeddings
This script visualizes word embeddings in 2D and demonstrates vector arithmetic using GloVe or Word2Vec.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from gensim.downloader import load as gensim_load

# Load a small pre-trained word2vec model (for demo)
print("Loading pre-trained word vectors (this may take a moment)...")
model = gensim_load('glove-wiki-gigaword-50')  # 50-dimensional GloVe vectors

# Select a set of words to visualize
words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'fruit', 'paris', 'france', 'london', 'england', 'dog', 'cat', 'animal', 'car', 'bus', 'vehicle']
embeddings = np.array([model[w] for w in words])

# Reduce to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10,7))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c='blue')
for i, word in enumerate(words):
    plt.text(embeddings_2d[i,0]+0.02, embeddings_2d[i,1]+0.02, word, fontsize=12)
plt.title('Word Embeddings Visualized (PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Demonstrate vector arithmetic: king - man + woman ≈ queen
vec = model['king'] - model['man'] + model['woman']
sims = model.similar_by_vector(vec, topn=5)
print("\nking - man + woman ≈ ?")
for word, score in sims:
    print(f"{word}: {score:.3f}")

# Visualize the arithmetic
words_arith = ['king', 'man', 'woman', 'queen']
emb_arith = np.array([model[w] for w in words_arith])
emb_arith_2d = pca.transform(emb_arith)
plt.figure(figsize=(7,6))
for i, word in enumerate(words_arith):
    plt.scatter(emb_arith_2d[i,0], emb_arith_2d[i,1], label=word)
    plt.text(emb_arith_2d[i,0]+0.01, emb_arith_2d[i,1]+0.01, word, fontsize=13)
# Draw arrows for king - man + woman = queen
plt.arrow(emb_arith_2d[0,0], emb_arith_2d[0,1], emb_arith_2d[1,0]-emb_arith_2d[0,0], emb_arith_2d[1,1]-emb_arith_2d[0,1], color='gray', head_width=0.05, length_includes_head=True)
plt.arrow(emb_arith_2d[1,0], emb_arith_2d[1,1], emb_arith_2d[2,0]-emb_arith_2d[1,0], emb_arith_2d[2,1]-emb_arith_2d[1,1], color='gray', head_width=0.05, length_includes_head=True)
plt.arrow(emb_arith_2d[2,0], emb_arith_2d[2,1], emb_arith_2d[3,0]-emb_arith_2d[2,0], emb_arith_2d[3,1]-emb_arith_2d[2,1], color='red', head_width=0.05, length_includes_head=True)
plt.title('Vector Arithmetic: king - man + woman ≈ queen')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 