import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Load your dataset (replace with your data)
data = pd.read_csv("your_dataset.csv")

# Train a Word2Vec model on your dataset
model = Word2Vec(data['words'].tolist(), min_count=1, size=100)

# Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Example usage
word1 = "cat"
word2 = "gato"  # Spanish for "cat"

if word1 in model.wv.vocab and word2 in model.wv.vocab:
    similarity = cosine_similarity(model.wv[word1], model.wv[word2])
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print("One or both words are not in the vocabulary.")
