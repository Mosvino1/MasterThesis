# MasterThesis
Analysis for my Master Thesis.

## 1) Overview
This repository contains code in Python and R Statistics to (a) prepare and clean thesis abstracts,
(b) extract per-document keyphrases, (c) train a BERTopic model and compute
yearly topic prevalence, and (d) produce descriptive plots, including paired
plots with Google Trends for visual context. Statistical alignment tests are
**not** used due to short annual series and sparse topic counts.

**Core outputs**
- Emerging/declining classification based on Δpp, growth ratios, and slopes
  (using 3-year centered moving averages).
- Yearly timelines for keyword themes and BERTopic topics.
- Annex figures for thesis–Google Trends paired plots (exploratory only).

Does not include webscraping and abstract extraction using LLama 3.1 due to sensitive Data in the code!  

## 3) R Preprocessing (Translation & Inspection)
Does not include cleaning (removing empty strings, duplicates etc.) 

-Descriptive Statistics.R
-Translation Abstracts.R
-Translation Titles.R

Adjust Filepaths and manipulated rows where appropriate. 

## 3) Python KeyBert
Required Columns for Input File: 
**Required columns**
- `year` (integer; 2015–2024)
- `programme` (string)
- `degree_level` (string: "BA" / "MA")
- `study_mode` (string: "VZ" / "BB")
- `title` (string)
- `abstract_en` (string; English abstract)

Run KeyBert.py

## 4) Python BerTopic


FILES TO RUN IN THIS ORDER:
├─ config.py                # Config file for File Paths and Model Parameters

├─ embed.py                 # BERTopic embeddings creation and saving

├─ fit.py                   # Create first run of the model for refinement

├─ merge.py                 # merge similiar topics

├─ label_and_visualize.py   # create final model



HELPERS:
├─ inspect.py               # helper for inspecting the model (run in console)

├─ Scientific Stopwords     # curated list of 


## 5) Environment
# Python
name: thesis-trends
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12.6
  - pip>=24.0
  - numpy=1.26.*
  - pandas=2.2.*
  - scikit-learn=1.5.*
  - umap-learn=0.5.*
  - hdbscan=0.8.*
  - matplotlib=3.9.*
  - plotly=5.*
  - tqdm=4.*
  - pyarrow
  - openpyxl
  - ipykernel
  - pip:
      - keybert==0.8.5
      - keyphrase-vectorizers==0.0.11
      - sentence-transformers==2.7.0
      - spacy==3.7.4
      - en-core-web-sm==3.7.1 # model wheel is optional; can install via spacy CLI
      - bertopic==0.16.0
      - nltk==3.9
      - pyLDAvis==3.4.1
      - rich==13.7.1


# R Statistic Packages: 
R version >= 4.3
Used Packages can be found in Code
