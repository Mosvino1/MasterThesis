import os, random, numpy as np, pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
from config import CSV_PATH, EMB_MODEL_NAME, EMB_PATH, SEED, ARTIFACTS

# Reproducibility (mostly matters later for UMAP; harmless here)
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

df = pd.read_csv(CSV_PATH, sep=";")
docs = df["combined_text"].astype(str).tolist()
print(f"Loaded {len(docs)} documents.")

if EMB_PATH.exists():
    print(f"Embeddings already found at {EMB_PATH}. Skipping re-encode.")
else:
    print(f"Encoding with SentenceTransformer: {EMB_MODEL_NAME}")
    st = SentenceTransformer(EMB_MODEL_NAME)
    embeddings = st.encode(docs, show_progress_bar=True)
    print("Embeddings shape:", getattr(embeddings, "shape", (len(embeddings), "unknown")))
    with open(EMB_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    print("Saved embeddings →", EMB_PATH)
