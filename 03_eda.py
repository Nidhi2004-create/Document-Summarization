"""
Task 2 - Exploratory Data Analysis (document length, word frequency, categories).
Heavy aggregation runs in Spark; only small result tables are moved to pandas for plotting.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from common import get_spark, RAW_CSV, CLEAN_PQ, OUT_DIR

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})
C1, C2, C3 = "#2E5EAA", "#E07A1F", "#3A9D6E"
spark = get_spark("EDA")

raw = (spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(RAW_CSV)
       .withColumn("n_words", F.size(F.split("text", r"\s+"))))
df = spark.read.parquet(CLEAN_PQ)

# ---- Fig 1: raw document length distribution (justifies the >=80 word filter) ---
raw_w = raw.select("n_words").toPandas()["n_words"].clip(upper=1500)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.hist(raw_w, bins=50, color=C1, edgecolor="white")
ax.axvline(80, color=C2, ls="--", lw=1.5); ax.text(95, ax.get_ylim()[1]*0.85, "80-word cut-off", color=C2)
ax.set_title("Raw corpus: words per document"); ax.set_xlabel("Words (clipped at 1500)"); ax.set_ylabel("Documents")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig1_raw_length.png"); plt.close(fig)

# ---- Fig 2: cleaned length distribution --------------------------------------------
p = df.select("n_words", "n_sentences", "avg_sent_len", "category").toPandas()
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.hist(p.n_words.clip(upper=1000), bins=40, color=C3, edgecolor="white")
ax.axvline(p.n_words.median(), color=C2, ls="--", lw=1.5)
ax.text(p.n_words.median()+20, ax.get_ylim()[1]*0.85, f"median = {int(p.n_words.median())}", color=C2)
ax.set_title("Cleaned corpus: words per document"); ax.set_xlabel("Words (clipped at 1000)"); ax.set_ylabel("Documents")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig2_clean_length.png"); plt.close(fig)

# ---- Fig 3: sentences per document ---------------------------------------------------
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.hist(p.n_sentences.clip(upper=50), bins=range(5, 52, 2), color=C1, edgecolor="white")
ax.set_title("Sentences per document"); ax.set_xlabel("Sentences (clipped at 50)"); ax.set_ylabel("Documents")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig3_sentences.png"); plt.close(fig)

# ---- Fig 4: top-20 word frequencies (Spark explode + groupBy) --------------------------
wf = (df.select(F.explode("tokens").alias("word")).groupBy("word").count()
        .orderBy(F.desc("count")))
top = wf.limit(20).toPandas().iloc[::-1]
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.barh(top.word, top["count"], color=C2)
ax.set_title("Top 20 words (stop-words removed)"); ax.set_xlabel("Frequency"); ax.xaxis.set_major_locator(plt.MaxNLocator(4))
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig4_top_words.png"); plt.close(fig)

# ---- Fig 5: Zipf's law (rank vs frequency, log-log) -----------------------------------
z = wf.toPandas(); z["rank"] = np.arange(1, len(z)+1)
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.loglog(z["rank"], z["count"], color=C1, lw=1.5)
ax.set_title("Word frequency vs rank (Zipf's law)"); ax.set_xlabel("Rank (log)"); ax.set_ylabel("Frequency (log)")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig5_zipf.png"); plt.close(fig)

# ---- Fig 6: category distribution -----------------------------------------------------
cat = p.category.value_counts().head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.barh(cat.index, cat.values, color=C3)
ax.set_title("Top 10 topic categories"); ax.set_xlabel("Documents")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig6_categories.png"); plt.close(fig)

# ---- Fig 7: document length by category (boxplot) -------------------------------------
top8 = p.category.value_counts().head(8).index
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.boxplot([p[p.category == c].n_words.clip(upper=1000) for c in top8], tick_labels=top8,
           showfliers=False, patch_artist=True, boxprops=dict(facecolor="#BFD3F2"))
ax.set_title("Words per document by category"); ax.set_ylabel("Words"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig7_len_by_cat.png"); plt.close(fig)

# ---- Summary statistics table ---------------------------------------------------------
stats = {
    "Documents in raw corpus": raw.count(),
    "Documents after filtering": len(p),
    "Total tokens (after stop-word removal)": df.agg(F.sum("n_tokens_clean")).first()[0],
    "Vocabulary size (unique tokens)": wf.count(),
    "Mean words / doc": round(p.n_words.mean(), 1),
    "Median words / doc": float(p.n_words.median()),
    "Max words / doc": int(p.n_words.max()),
    "Mean sentences / doc": round(p.n_sentences.mean(), 1),
    "Mean words / sentence": round(p.avg_sent_len.mean(), 1),
    "Distinct categories": p.category.nunique(),
}
s = pd.Series(stats, name="value"); s.to_csv(f"{OUT_DIR}/eda_stats.csv")
print("=== EDA summary statistics ==="); print(s.to_string())
print("\nTop 10 words:"); print(wf.limit(10).toPandas().to_string(index=False))
print("\nFigures saved to", OUT_DIR)
spark.stop()
