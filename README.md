# Movie Review Summaries + Semantic Recommender (Transformers)

**Notebook:** `MovieRecommenderTransformers.ipynb`  •  **Data:** `Movies_27K_Reviews.csv`  •  **Models:** summarization `t5-small`, sentiment `distilbert-base-uncased-finetuned-sst-2-english`, embeddings `sentence-transformers/all-MiniLM-L6-v2`

This project builds **concise summaries** of both **critic** and **audience** reviews for thousands of movies, then turns those summaries into a **semantic movie recommender** you can query in natural language.

- **Summarization (T5):** distills multiple lines of reviews per film into a short, readable synopsis.
- **Quality & Sentiment:** checks summary faithfulness via **TF‑IDF cosine similarity** and correlates **LLM/VADER sentiment** with critic/audience scores.
- **Semantic search & recs (SBERT):** embeds summaries with `all-MiniLM-L6-v2` and retrieves movies via cosine similarity.

---

## ✨ What’s in this repo
- A reproducible notebook that:
  - Cleans and canonicalizes **studio names** (e.g., Columbia/Sony, Disney, Paramount…)
  - De‑duplicates review lines per **(studio, movie_name)**
  - Generates **critic_summary** and **audience_summary** with a GPU‑aware T5 pipeline
  - Evaluates summary **similarity** to source reviews (TF‑IDF cosine)
  - Scores summaries with **LLM sentiment** (DistilBERT SST‑2) and **VADER**, correlating with real ratings
  - Builds a **content‑based recommender** from sentence embeddings
  - Exports ready‑to‑use artifacts:
    - `processed/movies_cleaned_summaries.csv`
    - `processed/critic_embeddings.npy`
    - `processed/aud_embeddings.npy`

---

## 🧱 Data
Place **`Movies_27K_Reviews.csv`** in the project root. Expected columns (normalized in the notebook):
`studio, rating, genre, movie_name, critic_score, audience_score, critic_line, audience_review`.

> The notebook maps common studio aliases to canonical labels, filters out rows without a known studio, and **keeps one aggregated record per film** (concatenating multiple critic/audience lines before summarization).

---

## 🛠️ Pipeline

### 1) Summarization
- **Model:** `t5-small` via `transformers.pipeline("summarization")`
- **Settings:** `do_sample=False`, `truncation=True`, `max_length≈1.6×max_words`, `min_length≈0.4×max_words` (default `max_words=150`)
- **Device:** auto‑detects GPU (`device_id=0`) or CPU (`-1`)

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=device_id)

def summarise(text, max_words=150):
    out = summarizer(text[:4000].strip(),
                     max_length=int(max_words*1.6),
                     min_length=int(max_words*0.4),
                     do_sample=False, truncation=True)[0]["summary_text"]
    return out
```

### 2) Summary quality (faithfulness proxy)
- **TF‑IDF → cosine similarity** between the aggregated raw text and the generated summary.
- Mean similarity in this dataset: **critic 0.260**, **audience 0.384**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(stop_words="english", max_features=20_000)

def sim_quality(orig, summ):
    X = vec.fit_transform([orig, summ])
    return cosine_similarity(X[0], X[1])[0,0]
```

### 3) Sentiment ↔ ratings
- **LLM sentiment:** `distilbert-base-uncased-finetuned-sst-2-english` via `pipeline("sentiment-analysis")`
- **Rule‑based sentiment:** **VADER**
- Correlations (r, p-value):
- **Critic (LLM) vs critic_score:** r=0.436, p=1.18e-72
- **Critic (VADER) vs critic_score:** r=0.229, p=7.79e-20
- **Audience (LLM) vs audience_score:** r=0.312, p=1.10e-40
- **Audience (VADER) vs audience_score:** r=0.191, p=9.12e-16

### 4) Semantic embeddings + recommender
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (`SentenceTransformer`)
- **Vectors:** critic and audience summaries → two embedding matrices
- **Retrieval:** cosine similarity vs query embedding

```python
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device_id)

def recommend(query, k=3):
    q = embedder.encode([query], convert_to_tensor=True).cpu()
    crit_sim = util.cos_sim(q, critic_mat)[0].numpy()
    aud_sim  = util.cos_sim(q,   aud_mat)[0].numpy()
    top_crit = df_sum.iloc[crit_sim.argsort()[-k:][::-1]][["movie","studio","genre"]]
    top_aud  = df_sum.iloc[aud_sim.argsort()[-k:][::-1]][["movie","studio","genre"]]
    return top_crit, top_aud
```

---

## 🚀 Quickstart

```bash
pip install -U torch transformers sentence-transformers scikit-learn pandas numpy vaderSentiment scipy
# optional (used in the notebook)
python -m spacy download en_core_web_sm -q
```

1. Put **`Movies_27K_Reviews.csv`** in the repo root.  
2. Run the notebook cells (GPU recommended for speed).  
3. Use `recommend("slow-burn atmospheric horror", k=3)` to get critic/audience picks.  
4. Artifacts will be created under `processed/`.

---

## 📊 Results (sample highlights)
- Summary faithfulness (TF‑IDF cosine, ↑ better): **critic 0.260**, **audience 0.384**.
- Sentiment–rating alignment (Pearson’s r):
  • Critic (LLM) vs critic_score: r=0.436 (p=p=1.18e-72)
  • Critic (VADER) vs critic_score: r=0.229 (p=p=7.79e-20)
  • Audience (LLM) vs audience_score: r=0.312 (p=p=1.10e-40)
  • Audience (VADER) vs audience_score: r=0.191 (p=p=9.12e-16)

> Audience summaries tend to be **closer to their source** (higher cosine), while critic summaries are more condensed. LLM sentiment correlates more strongly with scores than VADER, as expected.

---

## 📁 Repo structure
```
.
├─ MovieRecommenderTransformers.ipynb
├─ processed/
│  ├─ movies_cleaned_summaries.csv
│  ├─ critic_embeddings.npy
│  └─ aud_embeddings.npy
└─ README.md
```

---

## 🧭 Next steps
- Try stronger summarizers (`facebook/bart-large-cnn`, `google/pegasus-xsum`) and add **ROUGE/BERTScore**.
- Add a simple **Streamlit** UI for querying and browsing results.
- Blend critic/audience channels (weighted fusion) and **calibrate** against real user watchlists.
