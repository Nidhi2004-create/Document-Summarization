"""
Task 3 - NLP document-summarisation models (extractive), built with PySpark / Spark ML.

Models compared
  M1  Lead-3            : first three sentences (classic news baseline)
  M2  Spark TF-IDF      : sentence score = length-normalised sum of document TF-IDF weights x position weight
  M3  Spark ML GBT      : supervised sentence-importance classifier (Gradient-Boosted Trees)
  M4  TextRank          : graph-based ranking on sentence-similarity (scikit-learn/numpy baseline)

Evaluation (Reuters has no human summaries, so proxy metrics are used - see report):
  ROUGE-1/2/L recall against the article headline, key-term coverage (top-10 TF-IDF terms kept),
  compression ratio, and runtime.  Everything is reported on the official Reuters TEST split.
"""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from common import get_spark, CLEAN_PQ, SUMM_PQ, OUT_DIR

K = 3                                    # sentences per summary
spark = get_spark("Summarizer")
docs = spark.read.parquet(CLEAN_PQ).select("doc_id", "category", "title", "sentences", "tokens", "n_words", "n_sentences")
N = docs.count()
t0 = time.time()

# ------------------------------------------------------------------ sentence table
sent = (docs.select("doc_id", "n_sentences", F.posexplode("sentences").alias("sent_idx", "sentence")))
tok = RegexTokenizer(inputCol="sentence", outputCol="w", pattern="[^a-zA-Z]+", minTokenLength=2)
sw = StopWordsRemover(inputCol="w", outputCol="terms")
sent = sw.transform(tok.transform(sent)).drop("w") \
         .withColumn("s_words", F.size(F.split("sentence", " "))).cache()

# ------------------------------------------------------------------ document TF-IDF
tf = docs.select("doc_id", F.explode("tokens").alias("term")).groupBy("doc_id", "term").agg(F.count("*").alias("tf"))
dfreq = tf.groupBy("term").agg(F.countDistinct("doc_id").alias("df"))
idf = dfreq.withColumn("idf", F.log((F.lit(N) + 1) / (F.col("df") + 1)) + 1).select("term", "idf")
w = tf.join(idf, "term").withColumn("w", F.col("tf") * F.col("idf")).select("doc_id", "term", "w").cache()

# ------------------------------------------------------------------ M2: TF-IDF sentence scoring
st = sent.select("doc_id", "sent_idx", F.explode("terms").alias("term"))
score = (st.join(w, ["doc_id", "term"], "left").groupBy("doc_id", "sent_idx")
           .agg(F.sum(F.coalesce("w", F.lit(0.0))).alias("wsum")))
sent = sent.join(score, ["doc_id", "sent_idx"], "left").fillna({"wsum": 0.0})
sent = (sent.withColumn("tfidf_norm", F.col("wsum") / F.sqrt(F.greatest(F.size("terms"), F.lit(1)).cast("double")))
            .withColumn("pos_weight", 1 + 1 / (1 + F.col("sent_idx")))
            .withColumn("tfidf_score", F.col("tfidf_norm") * F.col("pos_weight")))

def top_k(df, score_col, name):
    win = Window.partitionBy("doc_id").orderBy(F.desc(score_col))
    sel = df.withColumn("rk", F.row_number().over(win)).filter(F.col("rk") <= K)
    return (sel.groupBy("doc_id")
              .agg(F.concat_ws(" ", F.expr("transform(array_sort(collect_list(struct(sent_idx, sentence))), x -> x.sentence)")).alias(name)))

lead3 = (sent.filter(F.col("sent_idx") < K).groupBy("doc_id")
             .agg(F.concat_ws(" ", F.expr("transform(array_sort(collect_list(struct(sent_idx, sentence))), x -> x.sentence)")).alias("lead3")))
tfidf_sum = top_k(sent, "tfidf_score", "tfidf_summary")

# ------------------------------------------------------------------ M3: supervised Spark ML (GBT)
# Distant supervision: label = the 2 sentences that share most words with the headline.
title_tok = RegexTokenizer(inputCol="title", outputCol="tw", pattern="[^a-zA-Z]+", minTokenLength=2)
title_sw = StopWordsRemover(inputCol="tw", outputCol="title_terms")
tt = title_sw.transform(title_tok.transform(docs.select("doc_id", "title"))).select("doc_id", "title_terms")
sent = sent.join(tt, "doc_id")
sent = sent.withColumn("overlap", F.size(F.array_intersect("terms", "title_terms")).cast("double"))
wl = Window.partitionBy("doc_id").orderBy(F.desc("overlap"), "sent_idx")
sent = sent.withColumn("ov_rank", F.row_number().over(wl)) \
           .withColumn("label", ((F.col("ov_rank") <= 2) & (F.col("overlap") > 0)).cast("double"))
# features (headline overlap is deliberately NOT a feature -> no label leakage)
sent = (sent.withColumn("rel_pos", F.col("sent_idx") / F.greatest(F.col("n_sentences") - 1, F.lit(1)))
            .withColumn("n_digits", F.length(F.regexp_replace("sentence", r"[^0-9]", "")).cast("double"))
            .withColumn("cap_ratio", F.size(F.expr("filter(split(sentence,' '), x -> x rlike '^[A-Z]')")) / F.col("s_words"))
            .withColumn("tfidf_pct", F.percent_rank().over(Window.partitionBy("doc_id").orderBy("tfidf_norm"))))
feat_cols = ["sent_idx", "rel_pos", "n_sentences", "s_words", "tfidf_norm", "tfidf_pct", "n_digits", "cap_ratio"]
sent = VectorAssembler(inputCols=feat_cols, outputCol="features").transform(sent).cache()

is_train = F.col("doc_id").startswith("training")
train, test = sent.filter(is_train), sent.filter(~is_train)
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=40, maxDepth=4, seed=42)
model = gbt.fit(train)
pred = model.transform(test).withColumn("prob", vector_to_array("probability")[1])
auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
imp = sorted(zip(feat_cols, model.featureImportances.toArray()), key=lambda x: -x[1])
print(f"\n[M3] GBT sentence-importance model  |  test AUC = {auc:.3f}")
print("Feature importances:", ", ".join(f"{n}={v:.2f}" for n, v in imp))
pd.DataFrame(imp, columns=["feature", "importance"]).to_csv(f"{OUT_DIR}/gbt_feature_importance.csv", index=False)
open(f"{OUT_DIR}/gbt_auc.txt", "w").write(f"{auc:.4f}")
gbt_sum = top_k(pred.select("doc_id", "sent_idx", "sentence", F.col("prob").alias("p")), "p", "gbt_summary")

# ------------------------------------------------------------------ assemble + save
final = (docs.select("doc_id", "category", "title", "n_words")
         .join(lead3, "doc_id", "left").join(tfidf_sum, "doc_id", "left")
         .join(gbt_sum, "doc_id", "left"))
final.write.mode("overwrite").parquet(SUMM_PQ)
spark_time = time.time() - t0
print(f"Spark pipeline (M1-M3, all documents) finished in {spark_time:.1f}s")

# key terms per document (top-10 TF-IDF) for coverage metric
kw = (w.withColumn("r", F.row_number().over(Window.partitionBy("doc_id").orderBy(F.desc("w"))))
        .filter("r <= 10").groupBy("doc_id").agg(F.collect_list("term").alias("keywords")))
pdf = (spark.read.parquet(SUMM_PQ).join(kw, "doc_id")
       .join(docs.select("doc_id", "sentences"), "doc_id")
       .filter(~is_train).toPandas())
print(f"Evaluation set: {len(pdf):,} test documents")

# ------------------------------------------------------------------ M4: TextRank (baseline)
from sklearn.feature_extraction.text import TfidfVectorizer
def textrank(sentences, k=K, d=0.85, iters=50):
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:k])
    S = (X @ X.T).toarray(); np.fill_diagonal(S, 0)
    rs = S.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    P = S / rs; r = np.ones(len(sentences)) / len(sentences)
    for _ in range(iters): r = (1 - d) / len(sentences) + d * P.T @ r
    idx = sorted(np.argsort(-r)[:k]); return " ".join(sentences[i] for i in idx)
t1 = time.time()
pdf["textrank_summary"] = [textrank(list(s)) for s in pdf.sentences]
tr_time = time.time() - t1

# ------------------------------------------------------------------ evaluation
from rouge_score import rouge_scorer
rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
import re
stop = set(StopWordsRemover.loadDefaultStopWords("english"))
def kw_cov(summary, keywords):
    toks = set(re.findall(r"[a-z]{2,}", summary.lower())); return np.mean([k in toks for k in keywords])

rows = []
for name, col in [("Lead-3", "lead3"), ("Spark TF-IDF", "tfidf_summary"), ("Spark ML GBT", "gbt_summary"), ("TextRank", "textrank_summary")]:
    r1 = r2 = rl = cov = comp = wc = 0
    for _, r in pdf.iterrows():
        sc = rs.score(r.title, r[col])
        r1 += sc["rouge1"].recall; r2 += sc["rouge2"].recall; rl += sc["rougeL"].recall
        cov += kw_cov(r[col], r.keywords); n = len(r[col].split()); wc += n; comp += n / r.n_words
    m = len(pdf)
    rows.append(dict(Model=name, ROUGE1_R=r1/m, ROUGE2_R=r2/m, ROUGEL_R=rl/m, KeyTermCoverage=cov/m,
                     AvgSummaryWords=wc/m, CompressionRatio=comp/m))
res = pd.DataFrame(rows).round(3)
res.to_csv(f"{OUT_DIR}/model_results.csv", index=False)
print("\n=== Model comparison (Reuters TEST split) ==="); print(res.to_string(index=False))
print(f"Runtime: Spark M1-M3 on {N:,} docs = {spark_time:.1f}s | TextRank on {len(pdf):,} docs = {tr_time:.1f}s")
pdf.drop(columns=["sentences"]).to_pickle(f"{OUT_DIR}/test_summaries.pkl")

# ------------------------------------------------------------------ chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
x = np.arange(len(res)); bw = 0.27
for i, (c, lab, col) in enumerate([("ROUGE1_R", "ROUGE-1 recall", "#2E5EAA"), ("ROUGE2_R", "ROUGE-2 recall", "#E07A1F"), ("ROUGEL_R", "ROUGE-L recall", "#3A9D6E")]):
    axs[0].bar(x + (i-1)*bw, res[c], bw, label=lab, color=col)
axs[0].set_xticks(x); axs[0].set_xticklabels(res.Model, rotation=20, ha="right"); axs[0].set_title("ROUGE vs. headline"); axs[0].legend(fontsize=7)
axs[1].bar(res.Model, res.KeyTermCoverage, color="#8B5CF6"); axs[1].set_title("Key-term coverage (top-10 TF-IDF)")
axs[1].set_xticklabels(res.Model, rotation=20, ha="right"); axs[1].set_ylim(0, 1)
for i, v in enumerate(res.KeyTermCoverage): axs[1].text(i, v + .02, f"{v:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig8_model_comparison.png", dpi=150)
spark.stop()
