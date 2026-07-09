#trained AI model
from sentence_transformers import SentenceTransformer
import numpy as np

#load model
model=SentenceTransformer("all-MiniLM-L6-v2")

sentence="How do i reset my password"
embedding=model.encode(sentence)
#384-dimensional vector

print(type(embedding))
print(embedding.shape)
print(embedding[:5])

def cosine_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

e1 = model.encode("How do I reset my password?")
e2 = model.encode("Account recovery procedure")
e3 = model.encode("Best restaurants in Mumbai")

print("\nSimilarity Scores:")
print("Password vs Recovery:", cosine_sim(e1, e2))
print("Password vs Restaurant:", cosine_sim(e1, e3))