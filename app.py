import os
import io
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from sklearn.decomposition import PCA

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
image_folder = "static/coco_images_resized"
upload_folder = "static/uploaded"
os.makedirs(upload_folder, exist_ok=True)

# Load the model and preprocess
model_name = "ViT-B-32"  # change from "ViT-B/32" to "ViT-B-32"
pretrained = "openai"

model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# Use the same model_name when getting tokenizer
tokenizer = get_tokenizer(model_name)


# Load embeddings
df = pd.read_pickle("image_embeddings.pickle")
image_embeddings = np.stack(df["embedding"].values)
file_names = df["file_name"].values

# Normalize embeddings
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

app = Flask(__name__)

def embed_text(text):
    text_tokens = tokenizer([text])
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb = F.normalize(text_emb, p=2, dim=1).cpu().numpy()
    return text_emb

def embed_image(pil_image):
    img_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = F.normalize(img_emb, p=2, dim=1).cpu().numpy()
    return img_emb

def compute_combined_embedding(text_emb, img_emb, lam):
    combined = lam * text_emb + (1.0 - lam) * img_emb
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    return combined

def top_k_similar(query_emb, k=5, use_pca=False, n_components=100):
    emb = image_embeddings
    q = query_emb
    if use_pca:
        pca = PCA(n_components=n_components)
        emb_pca = pca.fit_transform(emb)
        q_pca = pca.transform(q)
        sim = (q_pca @ emb_pca.T)[0]
    else:
        sim = (q @ emb.T)[0]

    top_indices = np.argsort(sim)[::-1][:k]
    top_scores = sim[top_indices]
    return top_indices, top_scores

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    text_query = request.form.get("text_query", "").strip()
    lam = request.form.get("lambda", "0.5")
    use_pca = request.form.get("use_pca", "off") == "on"
    n_components = request.form.get("n_components", "100")

    try:
        n_components = int(n_components)
    except:
        n_components = 100

    try:
        lam = float(lam)
    except:
        lam = 0.5

    image_file = request.files.get("image_file")
    text_provided = (text_query != "")
    image_provided = (image_file is not None and image_file.filename != "")

    if not text_provided and not image_provided:
        return "Please provide at least a text query or an image query."

    if text_provided:
        text_emb = embed_text(text_query)
    else:
        text_emb = None

    if image_provided:
        img_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(img_path)
        pil_img = Image.open(img_path).convert("RGB")
        img_emb = embed_image(pil_img)
    else:
        img_emb = None

    if text_emb is not None and img_emb is not None:
        query_emb = compute_combined_embedding(text_emb, img_emb, lam)
    elif text_emb is not None:
        query_emb = text_emb
    else:
        query_emb = img_emb

    top_indices, top_scores = top_k_similar(query_emb, k=5, use_pca=use_pca, n_components=n_components)

    results = []
    for idx, score in zip(top_indices, top_scores):
        results.append({
            "image_path": os.path.join("static/coco_images_resized", file_names[idx]),
            "score": f"{score:.4f}"
        })

    query_image_path = img_path if image_provided else None

    return render_template(
        "results.html",
        text_query=text_query,
        image_query=query_image_path,
        lambda_val=lam,
        use_pca=use_pca,
        n_components=n_components,
        results=results
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
