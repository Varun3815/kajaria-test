import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import faiss
import os
import pickle
from config import TILE_FOLDER

# Load model
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def extract_feature(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ Failed to open {image_path}: {e}")
        return np.zeros(512)

    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(tensor).squeeze().numpy()
    return feature.reshape(1, -1)


def build_faiss_index():
    index = faiss.IndexFlatL2(512)
    tile_map = []
    features = []

    for fname in os.listdir(TILE_FOLDER):
        path = os.path.join(TILE_FOLDER, fname)
        vec = extract_feature(path)
        if vec is not None:
            index.add(vec)
            tile_map.append(fname)
            features.append(vec)

    # Save index and filenames
    with open("tile_map.pkl", "wb") as f:
        pickle.dump(tile_map, f)
    faiss.write_index(index, "tiles.index")
    print(f"✅ Indexed {len(tile_map)} tiles")


def search_similar(image_path, top_k=5, threshold=0.6):
    index = faiss.read_index("tiles.index")
    with open("tile_map.pkl", "rb") as f:
        tile_map = pickle.load(f)

    query_vec = extract_feature(image_path)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)  # Convert L2 to similarity (optional)
        if similarity >= threshold:
            results.append((tile_map[idx], similarity))

    return results
