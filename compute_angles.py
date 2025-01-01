#!/usr/bin/env python3
# compute_angles.py

import os
import random
import string
import numpy as np
import torch
import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# Download NLTK data if not available
nltk.download('brown')
nltk.download('punkt')

MODEL_TO_LOAD = "../result/SumCSE_ortho-l0.1_m0.1"  # or "roberta-large", etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_LOAD)
model = AutoModel.from_pretrained(MODEL_TO_LOAD)
model.eval().to(DEVICE)

def encode_sentences(sentences, batch_size=32):
    """Encode a list of sentences into embeddings, normalized to unit vectors."""
    all_embs = []
    for start_idx in range(0, len(sentences), batch_size):
        batch_sents = sentences[start_idx : start_idx + batch_size]
        inputs = tokenizer(batch_sents, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Typical approach: use the [CLS] embedding or pooler_output
            # If using final hidden state's [CLS], we do:
            embeddings = outputs.last_hidden_state[:, 0, :]  # shape (batch, hidden_dim)
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embs.append(embeddings.cpu())
    return torch.cat(all_embs, dim=0).numpy()

def get_angles(embeddings):
    """Compute pairwise angles (in degrees) for a list of embeddings."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    dot_products = embeddings @ embeddings.T
    cosines = np.clip(dot_products / (norms * norms.T), -1.0, 1.0)
    # Only compute upper-triangular (i < j)
    angles = []
    for i in range(n):
        for j in range(i+1, n):
            angle_deg = np.degrees(np.arccos(cosines[i, j]))
            angles.append(angle_deg)
    return np.array(angles)

def create_random_meaningful_sentences(n):
    """Sample n random sentences from the Brown corpus, ensuring each has >= 5 tokens."""
    all_sents = brown.sents()
    sents = [' '.join(s) for s in all_sents if len(s) >= 5]
    random_sample = random.sample(sents, n)
    return random_sample

def main():
    n = 100
    # Generate random sentences
    sentences = create_random_meaningful_sentences(n)
    # Encode
    embeddings = encode_sentences(sentences)
    # Compute angles
    angles = get_angles(embeddings)
    mean_angle = np.mean(angles)
    var_angle = np.var(angles)

    print(f"\nComputed angles for {n} random Brown sentences.")
    print(f"Mean angle: {mean_angle:.2f} degrees")
    print(f"Variance: {var_angle:.2f} degrees")

    # Optional: histogram
    plt.hist(angles, bins=50, alpha=0.6)
    plt.title("Distribution of Angles Between Random Brown Sentence Embeddings")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    main()
