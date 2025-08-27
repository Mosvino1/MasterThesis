import os, random, numpy as np, re, pickle, json
import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

from config import (
    CSV_PATH, STOPWORDS_PATH, EMB_PATH,
    MODEL_BASE_DIR, OUTPUTS, FIGURES,
    SEED, NGRAM_RANGE, MIN_DF, MAX_DF,
    TOP_N_FOR_BARCHART, TOP_N_FOR_TIMELINE,
    DO_REDUCE_OUTLIERS_IN_BASE, OUTLIER_THRESHOLD
)

# ===== Load Select Topic Logic =====

def select_topics_from_summary(summary_df, mode, top_n, include_list, always_include):
    """Return an ordered list of topic IDs to display."""
    topics_available = summary_df["Topic"].tolist()
    chosen = []
    if mode == "include":
        chosen = [t for t in include_list if t in topics_available]
    elif mode == "top_n":
        chosen = summary_df.sort_values("Count", ascending=False)["Topic"].tolist()[: int(top_n)]
    else:
        chosen = summary_df["Topic"].tolist()

    # ensure uniqueness and keep order
    seen = set()
    chosen = [t for t in chosen if not (t in seen or seen.add(t))]

    # always-include at the end if present
    for t in always_include or []:
        if (t in topics_available) and (t not in chosen):
            chosen.append(t)
    return chosen

# ===== Reproducibility =====
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

# ===== Load data =====
df = pd.read_csv(CSV_PATH, sep=";")
docs = df["combined_text"].astype(str).tolist()
years = pd.to_numeric(df["year"], errors="coerce")

# ===== Stopwords =====
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    sci_stop = [w.strip() for w in f if w.strip()]
from nltk.corpus import stopwords
try:
    eng_stop = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download("stopwords")
    eng_stop = stopwords.words("english")

all_stop = sorted(set(eng_stop) | set(sci_stop))
phrase_stops = [s for s in all_stop if " " in s]
token_stops  = [s for s in all_stop if " " not in s]

# ===== Representation-only cleaning =====
docs_repr = [re.sub(r"-\s+", "", d) for d in docs]  # fix hyphen linebreaks
if phrase_stops:
    patt = re.compile(r"\b(" + "|".join(map(re.escape, phrase_stops)) + r")\b", flags=re.IGNORECASE)
    docs_repr = [patt.sub(" ", d) for d in docs_repr]

# ===== Load embeddings =====
with open(EMB_PATH, "rb") as f:
    embeddings = pickle.load(f)
assert len(embeddings) == len(docs), "Embeddings and docs length mismatch."

# ===== Seed topics (your list) =====
seed_topics = {
    "Green Topics": ["sustainability","sustainable","sustainably","csr","climate","environmental","environmentally",
                     "emission","emissions","ecosystem","ecosystems","renewable","ecology","green","greener",
                     "greenwashing","greenhouse","greening","co2","carbon","energy"],
    "AI & Automation": ["artificial intelligence","ai","machine learning","deep learning","automation","robot","robots",
                        "robotics","chatbot","chatbots","natural language processing","nlp","neural network","neural networks"],
    "Sales & Customer Marketing": ["marketing","geomarketing","neuromarketing","advertising","advertisement","advertised",
                                   "advertiser","advertisers","branding","rebranding","customer","customers","consumer",
                                   "consumers","consumerism","retail","retailing","retailers","sales","selling","merchandising"],
    "Supply Chains": ["supply chain","supply chains","supply network","value chain","production chain","logistics",
                      "logistic","intralogistics","procurement","delivery","distribution","distribution network","shipping",
                      "supplier","suppliers","transport","transportation","warehouse","warehouses","inventory"],
    "Social Media": ["social media","instagram","facebook","twitter","tiktok","influencer","influencers","youtube"],
    "Pandemic": ["covid","covid-19","covid 19","covid19","coronavirus","corona","pandemic","pandemics","epidemic","epidemics"],
    "Gender & Diversity": ["gender","genders","woman","women","female","females","male","males","diversity","equality",
                           "inequality","inclusion"],
    "Business Innovation": ["innovation","innovations","disruption","disruptions","creative","entrepreneur","entrepreneurs",
                            "entrepreneurial","entrepreneurship","startup","startups","start-up","start-ups",
                            "digital transformation","design thinking"],
    "Employment": ["employment","underemployment","unemployment","employee","employees","jobs","jobsharing","labour",
                   "labor","staff","staffing","workforce","workforces","workplace","workplaces"],
    "Education": ["education","educational","reeducation","curriculum","online learning","e-learning","elearning",
                  "learning motivation","distance learning","organizational learning","learning types","virtual learning",
                  "learning development","student","students","school","schools","schooling","schoolchildren",
                  "teaching","training","trainings"],
    "Generation": ["generation","generational","generationen","generations","intergenerational","multigenerational",
                   "millennial","millennials","boomer","boomers","cohort","cohorts","demographic","demographics",
                   "demographical","sociodemographic","gen z","generation z","thegeneration"],
    "International Trade & Policy": ["international trade","policy","policies","eu","eu27","eur","euro","eurasia","europe",
                                     "european","global","globalisation","globalization","foreign","regulation","regulations",
                                     "regulator","regulators","tariff","tariffs","sanction","sanctions","international"],
    "Investment & Finance": ["investment","investments","finance","financial","financing","financed","investor","investors",
                             "capital","capitalization","equity","crowdfunding","crowdfinancing","credit","credits",
                             "creditworthiness","loan","loans","bank","banks","banking","bankrupt","bankruptcy"],
    "Financial Crime & Corruption": ["corruption","crime","fraud","fraudsters","fraudulent","ethic","ethical","ethics",
                                     "ethically","unethical"],
    "Leadership, Change & Agility": ["leadership","leaders","leader","leading","agile","agility","change process",
                                     "cultural change","change business","digital change","change corporate",
                                     "companies change","change brand","succession","change management","changemanagement",
                                     "changeover","innovation management","organisational","organizational","top management",
                                     "lean management","talent management","performance management","personnel management",
                                     "stakeholder management","management process"],
    "Emerging Technologies & Data": ["digital","digitalisation","digitalization","digitally","digitalized","digitalised",
                                     "blockchain","blockchains","analytics","iot","iiot","cloud","cryptocurrency",
                                     "cryptocurrencies","crypto","data protection","data management","datamanagement",
                                     "dataset","datasets","data security","telemedicine","virtual","virtual reality","vr"],
    "Mental Health": ["stress","lifestyle","emotion","psychological","mental","emotional","wellbeing","burnout",
                      "mindfulness","therapy","psychology"]
}
seed_topic_list = list(seed_topics.values())

# ===== Models =====
umap_model = UMAP(n_neighbors=35, n_components=10, min_dist=0.0, metric="cosine", random_state=SEED)
hdb_model  = HDBSCAN(min_cluster_size=20, min_samples=5, metric="euclidean",
                     cluster_selection_method="leaf", prediction_data=True)

vectorizer_model = CountVectorizer(
    stop_words=token_stops,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF
)

topic_model = BERTopic(
    seed_topic_list=seed_topic_list,
    umap_model=umap_model,
    hdbscan_model=hdb_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

# ===== Fit =====
print("Fitting BERTopic...")
topics, probabilities = topic_model.fit_transform(docs_repr, embeddings=embeddings)
n_topics = len(set([t for t in topics if t != -1]))
print(f"Found {n_topics} topics (excluding -1).")


# ===== Assign Outlier Topics =====
if DO_REDUCE_OUTLIERS_IN_BASE:
    print(f"Reducing outliers in BASE with threshold={OUTLIER_THRESHOLD}")
    new_topics = topic_model.reduce_outliers(
        documents=docs_repr,  # <-- correct keyword
        topics=topics,
        probabilities=probabilities,
        strategy="probabilities",
        threshold=OUTLIER_THRESHOLD
    )
    topic_model.update_topics(docs_repr, topics=new_topics, vectorizer_model=vectorizer_model)
    topics = new_topics
    # (tidy) recompute for consistency
    topics, probabilities = topic_model.transform(docs_repr, embeddings=embeddings)

# ===== Save base model =====
topic_model.save(str(MODEL_BASE_DIR))
print("Saved model →", MODEL_BASE_DIR)

# ===== Exports =====
# Assignments
doc_topics = pd.DataFrame({
    "Document_ID": range(len(topics)),
    "Year": df["year"],
    "Assigned_Topic": topics
})
# quick keywords column
kw = {t: (", ".join([w for w,_ in topic_model.get_topic(t)][:3]) if t != -1 else "Outlier")
      for t in set(topics)}
doc_topics["Topic_Keywords"] = doc_topics["Assigned_Topic"].map(kw)
doc_topics.to_csv(OUTPUTS / "thesis_topic_assignments_base.csv", index=False)

# Topic summary
summary = topic_model.get_topic_info()
summary.to_csv(OUTPUTS / "topic_summary_counts_base.csv", index=False)

# ===== Visuals =====
from config import (
    TOPICS_MODE_BASE, TOPICS_TOP_N_BASE, TOPICS_INCLUDE_BASE, TOPICS_ALWAYS_INCLUDE_BASE
)

# Choose topics for BASE visuals
base_topics = select_topics_from_summary(
    summary_df=summary,
    mode=TOPICS_MODE_BASE,
    top_n=TOPICS_TOP_N_BASE,
    include_list=TOPICS_INCLUDE_BASE,
    always_include=TOPICS_ALWAYS_INCLUDE_BASE,
)

# --- Topic Distance Map (BASE)
fig_topics_map_base = topic_model.visualize_topics(topics=base_topics)  # 2D distances of topics
fig_topics_map_base.write_html(str(FIGURES / "topic_distance_base.html"))

# --- Similarity Heatmap (BASE)
fig_heat_base = topic_model.visualize_heatmap(topics=base_topics)
fig_heat_base.write_html(str(FIGURES / "topic_heatmap_base.html"))

# Hierarchy
hier = topic_model.hierarchical_topics(docs_repr)
topic_model.visualize_hierarchy(hierarchical_topics=hier).write_html(str(FIGURES / "topic_hierarchy_base.html"))

print("Exports done → outputs/ & figures/")

# Catalog of topics with top words
from pathlib import Path
catalog_rows = []
for row in summary.itertuples(index=False):
    t = row.Topic
    if t == -1:
        continue
    words = [w for w, _ in topic_model.get_topic(t)]
    catalog_rows.append({
        "Topic": t,
        "Count": row.Count,
        "Name": row.Name,
        "TopWords_10": ", ".join(words[:10])
    })
pd.DataFrame(catalog_rows).to_csv(OUTPUTS / "topic_catalog_base.csv", index=False)

# Similarity suggestions between topics using c-TF-IDF cosine
from sklearn.metrics.pairwise import cosine_similarity

SIM_THRESHOLD = 0.65   # was 0.75; relax a bit to surface candidates

# 1) Decide which topics to compare (exclude -1 for similarity only)
topics_list = [t for t in summary["Topic"].tolist() if t != -1]

# 2) Map row order of c_tf_idf_ to the order of get_topics() keys (excluding -1)
valid_order = [t for t in topic_model.get_topics().keys() if t != -1]
row_index = {t: i for i, t in enumerate(valid_order)}

# Some topics may not be in valid_order after merges/reductions; filter safely
row_ids = [row_index[t] for t in topics_list if t in row_index]
if not row_ids or len(row_ids) < 2:
    pairs_df = pd.DataFrame(columns=["Topic_A", "Topic_B", "CosineSimilarity"])
else:
    ctfidf = topic_model.c_tf_idf_[row_ids, :]
    S = cosine_similarity(ctfidf)

    pairs = []
    for i in range(len(row_ids)):
        for j in range(i + 1, len(row_ids)):
            sim = float(S[i, j])
            if sim >= SIM_THRESHOLD:
                pairs.append({
                    "Topic_A": topics_list[i],
                    "Topic_B": topics_list[j],
                    "CosineSimilarity": round(sim, 4)
                })

    pairs_df = pd.DataFrame(pairs, columns=["Topic_A", "Topic_B", "CosineSimilarity"])

# 3) Write CSV
if pairs_df.empty:
    print(f"No topic pairs exceeded similarity threshold {SIM_THRESHOLD}. "
          f"Writing an empty merge_suggestions.csv so the pipeline continues.")
else:
    pairs_df = pairs_df.sort_values("CosineSimilarity", ascending=False)

pairs_df.to_csv(OUTPUTS / "merge_suggestions.csv", index=False)