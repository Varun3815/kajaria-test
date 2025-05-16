import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import joblib
from config import TILE_FOLDER

# Load ResNet-18 model and remove classifier layer
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image transform for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ Error loading {image_path}: {e}")
        return np.zeros(512)

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor).squeeze()
    return features.numpy()

def compute_hash(image_path):
    """Compute a simple average hash using Pillow + NumPy (compatible with latest Pillow)."""
    try:
        image = Image.open(image_path).convert('L').resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.array(image)
        avg = pixels.mean()
        bits = pixels > avg
        return ''.join(['1' if b else '0' for b in bits.flatten()])
    except Exception as e:
        print(f"⚠️ Could not hash image {image_path}: {e}")
        return None

def build_image_index():
    all_features = []
    tile_names = []
    seen_hashes = set()

    for fname in os.listdir(TILE_FOLDER):
        path = os.path.join(TILE_FOLDER, fname)
        image_hash = compute_hash(path)
        if not image_hash or image_hash in seen_hashes:
            continue
        seen_hashes.add(image_hash)

        feats = extract_features(path)
        if np.linalg.norm(feats) != 0:
            all_features.append(feats)
            tile_names.append(fname)

    if not all_features:
        print("❌ No valid images found to index.")
        return

    all_features = np.vstack(all_features)
    joblib.dump((tile_names, all_features), 'tile_index.pkl')
    print("✅ Feature index saved as 'tile_index.pkl' with duplicates removed.")

def find_best_matches(uploaded_image_path, top_k=5, min_threshold=0.6):
    if not os.path.exists("tile_index.pkl"):
        build_image_index()

    tile_names, feature_matrix = joblib.load("tile_index.pkl")
    uploaded_features = extract_features(uploaded_image_path).reshape(1, -1)

    model_nn = NearestNeighbors(n_neighbors=min(top_k, len(tile_names)), metric="cosine")
    model_nn.fit(feature_matrix)
    distances, indices = model_nn.kneighbors(uploaded_features)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 - dist
        if similarity >= min_threshold:
            results.append((tile_names[idx], similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)
