import pandas as pd
from pathlib import Path
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

# NEW: tqdm for a progress bar
from tqdm import tqdm
tqdm.pandas(desc="Extracting keywords")   # enables .progress_apply on pandas

# ---- Config ----
INPUT_CSV  = "df_clean_all_translated.csv"
OUTPUT_CSV = "theses_keywords_minimal.csv"
SEP        = ";"
NROWS      = None
TOP_N      = 10
NGRAMS     = (1,1)

def combine_title_abstract(title: str, abstract: str) -> str:
    return (title.strip() + ". " + abstract.strip()).strip(". ").strip()

def main():
    path = Path(INPUT_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path.resolve()}")

    df = pd.read_csv(path, sep=SEP, quoting=1, nrows=NROWS)

    # Ensure string dtype and no NaNs
    df[["topic_en", "abstracts_en"]] = (
        df[["topic_en", "abstracts_en"]].fillna("").astype(str)
    )

    combined_col = "combined_text"
    df[combined_col] = [
        combine_title_abstract(t, a)
        for t, a in zip(df["topic_en"], df["abstracts_en"])
    ]

    kw_model = KeyBERT()

    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=NGRAMS,
        min_df=1
    )

    def extract_keywords(text: str, topn: int = TOP_N):
        if not text or not text.strip():
            return []
        kws = kw_model.extract_keywords(
            text,
            vectorizer=vectorizer,
            keyphrase_ngram_range=NGRAMS,
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=topn
        )
        return [k for k, _ in kws]

    # PROGRESS BAR HERE 👇
    df["keywords_list"] = df[combined_col].progress_apply(extract_keywords)
    df["keywords_top10"] = df["keywords_list"].apply(lambda xs: ", ".join(xs))

    out = Path(OUTPUT_CSV)
    df.to_csv(out, sep=";", index=False, encoding="utf-8-sig")
    print(f"Saved: {out.resolve()}")
    print(df[["year", "topic_en", "keywords_top10"]].to_string(index=False))

if __name__ == "__main__":
    main()
