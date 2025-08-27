# 03_merge_minimal.py
import re, pickle
import pandas as pd
from pathlib import Path
from bertopic import BERTopic

from config import CSV_PATH, EMB_PATH, MODEL_BASE_DIR, OUTPUTS
try:
    from config import MODEL_MERGED_DIR
except Exception:
    MODEL_MERGED_DIR = Path("artifacts/bertopic_merged")

# 1) Your groups (first ID in each list survives)
topics_to_merge = [
    [0,73],
    [10,35],
    [12,15,36],
    [27,59],
    [24,67],
    [22,64],
    [20,53],
    [83,34],
    [41,76],
    [49,62],
    [16,42],
    [37,51],
    [21,48],
    [33,74],
    [2,57],
    [11,66],
]

# 2) Load docs used by BERTopic for merging
df = pd.read_csv(CSV_PATH, sep=";")
docs = df["combined_text"].astype(str).tolist()
docs_repr = [re.sub(r"-\s+", "", d) for d in docs]  # minimal cleanup

# (embeddings are not needed for merging, only if you later transform)
with open(EMB_PATH, "rb") as f:
    embeddings = pickle.load(f)

# 3) Load base model
topic_model = BERTopic.load(str(MODEL_BASE_DIR))
print("Loaded BASE model:", MODEL_BASE_DIR)

# 4) One-shot merge (list of lists)
topic_model.merge_topics(docs_repr, topics_to_merge)  # first ID in each sublist survives

# 5) Save merged model & export summary
MODEL_MERGED_DIR = Path(MODEL_MERGED_DIR)
MODEL_MERGED_DIR.parent.mkdir(parents=True, exist_ok=True)
topic_model.save(str(MODEL_MERGED_DIR))
print("Saved MERGED model →", MODEL_MERGED_DIR)

OUTPUTS = Path(OUTPUTS); OUTPUTS.mkdir(parents=True, exist_ok=True)
summary = topic_model.get_topic_info()
summary.to_csv(OUTPUTS / "topic_summary_counts_merged.csv", index=False)
print("Wrote →", OUTPUTS / "topic_summary_counts_merged.csv")
