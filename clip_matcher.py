# clip_matcher.py
import os
import torch
import open_clip
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device)
model.eval()

def encode_text(text):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features.cpu().numpy()

def search_tiles_by_text(query, top_k=20, min_threshold=0.2):
    if not os.path.exists("tile_clip_index.pkl"):
        print("❌ You must run reindex_clip.py first.")
        return []

    tile_names, feature_matrix = joblib.load("tile_clip_index.pkl")
    text_vector = encode_text(query)

    nn_model = NearestNeighbors(n_neighbors=min(top_k, len(tile_names)), metric="cosine")
    nn_model.fit(feature_matrix)

    distances, indices = nn_model.kneighbors(text_vector)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        sim = 1 - dist
        if sim >= min_threshold:
            results.append((tile_names[idx], sim))

    return sorted(results, key=lambda x: x[1], reverse=True)
