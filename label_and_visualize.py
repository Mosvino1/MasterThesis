import os, random, numpy as np, json, pickle, re
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from umap import UMAP

from config import (
    CSV_PATH, STOPWORDS_PATH, EMB_PATH,
    MODEL_MERGED_DIR, MODEL_FINAL_DIR,
    OUTPUTS, FIGURES,
    SEED,
    TOPICS_MODE_FINAL, TOPICS_TOP_N_FINAL, TOPICS_INCLUDE_FINAL, TOPICS_ALWAYS_INCLUDE_FINAL,
    TIMELINE_USE_SELECTED_TOPICS, TIMELINE_TOP_N,
    CLASS_TOPICS_MODE, CLASS_TOPICS_TOP_N, CLASS_TOPICS_INCLUDE, CLASS_TOPICS_ALWAYS_INCLUDE,
    DOCS_VIS_TITLES_COLS, DOCS_VIS_SAMPLE, DOCS_VIS_HIDE_ANNOTATIONS,
    INCLUDE_OUTLIER_IN_TOT, INCLUDE_OUTLIER_IN_TRENDS
)

# ------------------ reproducibility ------------------
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

# ------------------ helpers ------------------
def select_topics_from_summary(summary_df, mode, top_n, include_list, always_include):
    topics_available = summary_df["Topic"].tolist()
    if mode == "include":
        chosen = [t for t in include_list if t in topics_available]
    elif mode == "top_n":
        chosen = summary_df.sort_values("Count", ascending=False)["Topic"].tolist()[: int(top_n)]
    else:
        chosen = topics_available[:]
    seen = set()
    chosen = [t for t in chosen if not (t in seen or seen.add(t))]
    for t in (always_include or []):
        if (t in topics_available) and (t not in chosen):
            chosen.append(t)
    return chosen

def first_present(colnames, columns):
    for c in colnames:
        if c in columns:
            return c
    return None

def get_custom_label_map(model):
    """Return {topic_id: label} across BERTopic versions if labels were set."""
    lbl = getattr(model, "custom_labels_", None)
    if isinstance(lbl, dict) and lbl:
        return lbl
    if hasattr(model, "get_topic_labels"):
        try:
            got = model.get_topic_labels()
            if isinstance(got, dict):
                return got
            if isinstance(got, list):
                return {i: v for i, v in enumerate(got)}
        except Exception:
            pass
    return {}

# ------------------ load data ------------------
df = pd.read_csv(CSV_PATH, sep=";")
docs = df["combined_text"].astype(str).tolist()
years = pd.to_numeric(df["year"], errors="coerce")

# ------------------ preprocessing ------------------
# Load stopwords
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    sci_stop = [w.strip() for w in f if w.strip()]
from nltk.corpus import stopwords
try:
    eng_stop = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download("stopwords"); eng_stop = stopwords.words("english")
all_stop = sorted(set(eng_stop) | set(sci_stop))
phrase_stops = [s for s in all_stop if " " in s]

# hyphen fix + phrase removal
docs_repr = [re.sub(r"-\s+", "", d) for d in docs]
if phrase_stops:
    patt = re.compile(r"\b(" + "|".join(map(re.escape, phrase_stops)) + r")\b", flags=re.IGNORECASE)
    docs_repr = [patt.sub(" ", d) for d in docs_repr]

# time-filtered corpus (for topics_over_time)
mask = years.notna()
docs_time  = [d for d, m in zip(docs_repr, mask) if m]
years_time = years[mask].astype(int).tolist()
nbins = len(sorted(set(years_time))) if len(set(years_time)) > 0 else 1

# ------------------ load embeddings ------------------
with open(EMB_PATH, "rb") as f:
    embeddings = pickle.load(f)
emb = np.asarray(embeddings)

# ------------------ load model ------------------
topic_model = BERTopic.load(str(MODEL_MERGED_DIR))
print("Loaded MERGED model:", MODEL_MERGED_DIR)



# ------------------ apply & persist custom labels ------------------
labels_path = OUTPUTS / "topic_labels.json"
if labels_path.exists():
    with open(labels_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    label_map.setdefault(-1, "Outlier")
    topic_model.set_topic_labels(label_map)
    print("Applied labels from", labels_path)
else:
    print("No topic_labels.json found → proceed without labels (you can re-run after adding it).")

# Ensure model has predictions for THIS corpus
topics, probabilities = topic_model.transform(docs_repr, embeddings=emb)

# ------------------ R-READY EXPORTS ------------------
# A) Document-level assignments (+ probability)
info_df = topic_model.get_document_info(docs_repr)

prog_col  = first_present(["program","programme","Program","Programme"], df.columns)
level_col = first_present(["level","Level"], df.columns)
full_col  = first_present(["fulltime","Fulltime","full_time","Full_time"], df.columns)
title_col = first_present(DOCS_VIS_TITLES_COLS, df.columns)

# resolve labels consistently
custom_label_map = get_custom_label_map(topic_model)
summary_tmp = topic_model.get_topic_info()
name_map = {row.Topic: row.Name for row in summary_tmp.itertuples(index=False)}
def label_for_topic(t):
    if t in custom_label_map and custom_label_map[t]:
        return custom_label_map[t]
    return "Outlier" if t == -1 else name_map.get(t, f"Topic {t}")

doc_topics = pd.DataFrame({
    "Document_ID": range(len(docs_repr)),
    "Year": df["year"].astype("Int64"),
    "Assigned_Topic": info_df["Topic"].astype("Int64"),
    "Assigned_Prob": info_df["Probability"] if "Probability" in info_df.columns
                    else pd.Series([pd.NA]*len(info_df))
})
doc_topics["Assigned_Label"] = doc_topics["Assigned_Topic"].map(label_for_topic)
if prog_col:  doc_topics["Program"]  = df[prog_col].astype(str)
if level_col: doc_topics["Level"]    = df[level_col].astype(str)
if full_col:  doc_topics["FullTime"] = df[full_col].astype(str)
if title_col: doc_topics["Title"]    = df[title_col].astype(str)

top5 = {t: ", ".join([w for w,_ in topic_model.get_topic(t)][:5]) if t != -1 else "Outlier"
        for t in set(info_df["Topic"])}
doc_topics["Topic_Keywords"] = doc_topics["Assigned_Topic"].map(top5)
doc_topics.to_csv(OUTPUTS / "thesis_topic_assignments_final.csv", index=False)

# B) Topic summary (overall) with Label column
summary = topic_model.get_topic_info()
summary["Label"] = summary["Topic"].apply(label_for_topic)
summary.to_csv(OUTPUTS / "topic_summary_counts_final.csv", index=False)

# C) Topics over time
tot = topic_model.topics_over_time(
    docs_time, years_time,
    nr_bins=nbins,
    global_tuning=True, evolution_tuning=True
)
tot_df = tot.copy()
# robust Year
if "Timestamp" in tot_df.columns:
    try:
        tot_df["Year"] = pd.to_datetime(tot_df["Timestamp"]).dt.year
    except Exception:
        tot_df["Year"] = pd.to_numeric(tot_df["Timestamp"], errors="coerce").astype("Int64")
else:
    guess = [c for c in tot_df.columns if "year" in c.lower()]
    if guess:
        tot_df["Year"] = pd.to_numeric(tot_df[guess[0]], errors="coerce").astype("Int64")
    else:
        raise ValueError("Could not infer Year in topics_over_time result.")

freq_col = "Frequency" if "Frequency" in tot_df.columns else ("Count" if "Count" in tot_df.columns else None)
if freq_col is None:
    raise ValueError("topics_over_time result missing Frequency/Count column.")

tot_df["Label"] = tot_df["Topic"].map(label_for_topic)
total_per_year = (pd.to_numeric(df["year"], errors="coerce")
                  .dropna().astype(int).value_counts().rename_axis("Year").reset_index(name="TotalDocsYear"))
tot_df = tot_df.merge(total_per_year, on="Year", how="left")
tot_df["Share"] = tot_df[freq_col] / tot_df["TotalDocsYear"]

if not INCLUDE_OUTLIER_IN_TOT and (-1 in tot_df["Topic"].unique()):
    tot_df = tot_df[tot_df["Topic"] != -1]

tot_out = tot_df[["Topic", "Label", "Year", freq_col, "TotalDocsYear", "Share"]].rename(
    columns={freq_col: "Frequency"}
)
tot_out.to_csv(OUTPUTS / "topics_over_time_final.csv", index=False)

# D) Topic dictionary
dict_rows = []
for row in summary.itertuples(index=False):
    t = row.Topic
    words = [w for w, _ in topic_model.get_topic(t)]
    dict_rows.append({
        "Topic": t,
        "Label": row.Label,
        "TopWords_15": ", ".join(words[:15])
    })
pd.DataFrame(dict_rows).to_csv(OUTPUTS / "topic_dictionary_final.csv", index=False)

# E) Google Trends query template
def suggested_query(t, label, words):
    label = (label or "").strip()
    if label and not str(label).lower().startswith("topic "):
        return label
    return " ".join(words[:2]) if words else f"topic_{t}"

rows = []
topic_ids_for_trends = summary["Topic"].tolist()
if not INCLUDE_OUTLIER_IN_TRENDS and (-1 in topic_ids_for_trends):
    topic_ids_for_trends = [t for t in topic_ids_for_trends if t != -1]
for t in topic_ids_for_trends:
    words = [w for w, _ in topic_model.get_topic(t)]
    label = label_for_topic(t)
    rows.append({
        "Topic": t,
        "Label": label,
        "Suggested_Query": suggested_query(t, label, words)
    })
pd.DataFrame(rows).to_csv(OUTPUTS / "topic_trends_queries.csv", index=False)

print("R-ready files written:")
print(" - thesis_topic_assignments_final.csv  (doc→topic + prob + metadata)")
print(" - topics_over_time_final.csv          (topic-year Frequency + Share)")
print(" - topic_dictionary_final.csv          (Topic, Label, TopWords_15)")
print(" - topic_trends_queries.csv            (Topic, Label, Suggested_Query)")

# ------------------ save FINAL model (labels persist) ------------------
topic_model.save(str(MODEL_FINAL_DIR))
print("Saved FINAL model →", MODEL_FINAL_DIR)

# ------------------ VISUALS ------------------
# Safe topic list for heatmap/distance map
freq_df = topic_model.get_topic_freq()
valid_topic_ids = set(freq_df.loc[freq_df["Topic"] != -1, "Topic"].tolist())
def keep_valid(ids): return [t for t in ids if t in valid_topic_ids]
ordered_by_freq = freq_df.loc[freq_df["Topic"] != -1, "Topic"].tolist()
all_topics_final = [t for t in ordered_by_freq if t in keep_valid(summary["Topic"].tolist())]

# A) Distance map & similarity heatmap
if all_topics_final:
    fig_topics_map_final = topic_model.visualize_topics(topics=all_topics_final, custom_labels=True)
else:
    fig_topics_map_final = topic_model.visualize_topics(top_n_topics=len(valid_topic_ids) or 10, custom_labels=True)
fig_topics_map_final.write_html(str(FIGURES / "topic_distance_final.html"))

if all_topics_final:
    fig_heat_final = topic_model.visualize_heatmap(topics=all_topics_final, custom_labels=True)
else:
    fig_heat_final = topic_model.visualize_heatmap(top_n_topics=min(25, len(valid_topic_ids) or 10), custom_labels=True)
fig_heat_final.write_html(str(FIGURES / "topic_heatmap_final.html"))

# B) Topics over time (reuse 'tot')
final_selected = select_topics_from_summary(
    summary_df=summary,
    mode=TOPICS_MODE_FINAL,
    top_n=TOPICS_TOP_N_FINAL,
    include_list=TOPICS_INCLUDE_FINAL,
    always_include=TOPICS_ALWAYS_INCLUDE_FINAL,
)
final_selected = keep_valid(final_selected)

if TIMELINE_USE_SELECTED_TOPICS and final_selected:
    fig_time_final = topic_model.visualize_topics_over_time(tot, topics=final_selected, custom_labels=True)
else:
    fig_time_final = topic_model.visualize_topics_over_time(tot, top_n_topics=TIMELINE_TOP_N, custom_labels=True)
fig_time_final.write_html(str(FIGURES / "topics_over_time_final.html"))

# C) Per-class visuals
prog_col  = first_present(["program","programme","Program","Programme"], df.columns)
level_col = first_present(["level","Level"], df.columns)
full_col  = first_present(["fulltime","Fulltime","full_time","Full_time"], df.columns)

def class_topics_choice():
    if CLASS_TOPICS_MODE == "same_as_final":
        return final_selected
    return select_topics_from_summary(
        summary_df=summary,
        mode=CLASS_TOPICS_MODE,
        top_n=CLASS_TOPICS_TOP_N,
        include_list=CLASS_TOPICS_INCLUDE,
        always_include=CLASS_TOPICS_ALWAYS_INCLUDE,
    )

class_topics = keep_valid(class_topics_choice())

def topics_per_class_viz(colname, filename_prefix):
    if not colname:
        print(f"Column not found for {filename_prefix}; skipping.")
        return
    classes = df[colname].fillna("Unknown").astype(str).tolist()
    tpc = topic_model.topics_per_class(docs_repr, classes)
    fig = topic_model.visualize_topics_per_class(
        tpc, topics=class_topics if class_topics else None, custom_labels=True
    )
    fig.write_html(str(FIGURES / f"{filename_prefix}.html"))
    tpc.to_csv(OUTPUTS / f"{filename_prefix}.csv", index=False)

topics_per_class_viz(prog_col,  "topics_per_program_final")
topics_per_class_viz(level_col, "topics_per_level_final")
topics_per_class_viz(full_col,  "topics_per_fulltime_final")

# D) Documents scatter — robust, minimal first, then labeled
from umap import UMAP

if not docs_repr:
    raise ValueError("docs_repr is empty — cannot visualize.")
if len(docs_repr) != len(emb):
    raise ValueError(f"Embeddings length ({len(emb)}) != docs length ({len(docs_repr)}).")

# Ensure the model carries assignments for THESE docs (needed after .load() on some versions)
topics, probabilities = topic_model.transform(docs_repr, embeddings=emb)
topic_model.topics_ = topics
topic_model.probabilities_ = probabilities

# 2D with UMAP (official pattern)
umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=SEED)
reduced_embeddings = umap_model.fit_transform(emb)

# Drop any rows with NaN/Inf (avoid empty/blank plots due to invalid points)
finite_mask = np.isfinite(reduced_embeddings).all(axis=1)
if not finite_mask.all():
    print(f"[docs scatter] Dropping {np.count_nonzero(~finite_mask)} rows with non-finite 2D coords.")
docs_repr_clean = [d for d, keep in zip(docs_repr, finite_mask) if keep]
reduced_clean   = reduced_embeddings[finite_mask]

# --- 1) Minimal plot (no filters, no custom labels) -> should ALWAYS show points
fig_docs_min = topic_model.visualize_documents(
    docs_repr_clean,
    reduced_embeddings=reduced_clean
)
fig_docs_min.write_html(str(FIGURES / "documents_scatter_final_min.html"))

# --- 2) Labeled plot (once we know points render)
# Prefer non-outlier topics, but if all are -1, include -1 so you still see points
present = sorted(set(t for t in np.array(topics)[finite_mask] if t is not None))
non_outliers = [t for t in present if t != -1]
topics_for_plot = non_outliers if non_outliers else present

fig_docs_lbl = topic_model.visualize_documents(
    docs_repr_clean,
    reduced_embeddings=reduced_clean,
    topics=topics_for_plot,
    custom_labels=True,
    hide_annotations=DOCS_VIS_HIDE_ANNOTATIONS
)
fig_docs_lbl.write_html(str(FIGURES / "documents_scatter_final.html"))

# Diagnostics
u, c = np.unique(np.array(topics)[finite_mask], return_counts=True)
print("Assigned topic counts (finite only):", dict(zip(u.tolist(), c.tolist())))
print("Docs plotted:", len(docs_repr_clean), "Reduced shape:", reduced_clean.shape)
print("Plotted topics:", topics_for_plot[:10], "… total:", len(topics_for_plot))
