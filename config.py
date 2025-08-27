from pathlib import Path

# ===== Paths =====
CSV_PATH = Path("theses_keywords_minimal.csv")
STOPWORDS_PATH = Path("Scientific Stopwords.txt")

ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True)
OUTPUTS   = Path("outputs");   OUTPUTS.mkdir(exist_ok=True)
FIGURES   = Path("figures");   FIGURES.mkdir(exist_ok=True)

EMB_MODEL_NAME = "all-mpnet-base-v2"  # you set this in your code
EMB_PATH  = ARTIFACTS / "embeddings_mpnet.pkl"

MODEL_BASE_DIR   = ARTIFACTS / "bertopic_base"
MODEL_MERGED_DIR = ARTIFACTS / "bertopic_merged"
MODEL_FINAL_DIR  = ARTIFACTS / "bertopic_final"


# ===== Reproducibility =====
SEED = 42

# ===== Vectorizer options for topic representation =====
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.95

# ===== Over-time viz =====
TOP_N_FOR_BARCHART = 10
TOP_N_FOR_TIMELINE = 6

# ===== Hierarchical reduction target =====
TARGET_K = 45  # adjust if you want a specific final topic count


# ===== Reduce Outliers ======
# add:
DO_REDUCE_OUTLIERS_IN_BASE = True
OUTLIER_THRESHOLD = 0.02  # 0.015–0.03 works well


# === Topic selection for Visualizations (base model) ===
TOPICS_MODE_BASE = "top_n"       # "top_n" or "include"
TOPICS_TOP_N_BASE = 25           # used if mode == "top_n"
TOPICS_INCLUDE_BASE = []         # e.g., [40, -1, 3]
TOPICS_ALWAYS_INCLUDE_BASE = []  # always append these if present

# === Topic selection for Visualizations (final model) ===
TOPICS_MODE_FINAL = "top_n"      # "top_n" or "include"
TOPICS_TOP_N_FINAL = 66
TOPICS_INCLUDE_FINAL = []        # e.g., [40, -1]
TOPICS_ALWAYS_INCLUDE_FINAL = []

# === Topics-over-time (final model) ===
TIMELINE_USE_SELECTED_TOPICS = False  # if False, use top_n inside viz call
TIMELINE_TOP_N = 66                  # only used when not using explicit topics

# === Per-class (final only) ===
# Re-usable topic selection for per-class plots (default: same as final)
CLASS_TOPICS_MODE = "same_as_final"  # "same_as_final" | "top_n" | "include"
CLASS_TOPICS_TOP_N = 68
CLASS_TOPICS_INCLUDE = []
CLASS_TOPICS_ALWAYS_INCLUDE = []

# === Document visualization (final only) ===
DOCS_VIS_TITLES_COLS = ["topic_en"]  # in order of preference
DOCS_VIS_SAMPLE = None     # None for all docs; use a number to keep figure light
DOCS_VIS_HIDE_ANNOTATIONS = True

# R exports: whether to include the outlier topic -1
INCLUDE_OUTLIER_IN_TOT = True        # for topics_over_time CSV
INCLUDE_OUTLIER_IN_TRENDS = False    # for trends query template


