"""
Task 1/2 - Big-data preprocessing with PySpark.

Steps: load CSV -> clean text -> sentence split -> tokenise -> stop-word removal
       -> compute length features -> save as Parquet for EDA and modelling.
"""
from pyspark.sql import functions as F

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from common import get_spark, RAW_CSV, CLEAN_PQ

spark = get_spark("Preprocessing")

# ---------- 1. Load ------------------------------------------------------------
raw = (spark.read.option("header", True).option("multiLine", True)
       .option("escape", '"').csv(RAW_CSV))
print(f"Raw documents loaded : {raw.count():,}")
raw.printSchema()

# ---------- 2. Clean ----------------------------------------------------------
df = (raw.dropna(subset=["text"])
        .withColumn("text", F.regexp_replace("text", r"\s*Reuter\s*\x03?\s*$", ""))   # trailing 'Reuter'
        .withColumn("text", F.regexp_replace("text", r"\s*<[A-Z0-9.\-]{1,10}>", ""))         # drop <TICKER.N> tags
        .withColumn("text", F.regexp_replace("text", r"<([^<>]+)>", "$1"))             # unwrap <Company Name>
        .withColumn("text", F.trim(F.regexp_replace("text", r"\s+", " ")))
        .filter(F.length("text") > 0))

# ---------- 3. Sentence split (regex; native Spark split avoids UDF worker crash) ---
# Split on sentence-ending punctuation followed by whitespace and a capital letter.
# Using Spark built-in split() keeps everything in the JVM - no Python worker needed.
df = df.withColumn("sentences",
                   F.filter(
                       F.split("text", r"(?<=[a-z0-9\"\)][.!?])\s+(?=[A-Z\"\(])"),
                       lambda s: F.length(s) > 0
                   )) \
       .withColumn("n_sentences", F.size("sentences"))

# ---------- 4. Tokenise + stop-words -------------------------------------------
tok = RegexTokenizer(inputCol="text", outputCol="tokens_raw", pattern="[^a-zA-Z]+",
                     minTokenLength=2, toLowercase=True)
sw = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens")
df = sw.transform(tok.transform(df))

# ---------- 5. Length features -------------------------------------------------
df = (df.withColumn("n_chars", F.length("text"))
        .withColumn("n_words", F.size(F.split("text", " ")))
        .withColumn("n_tokens_clean", F.size("tokens"))
        .withColumn("avg_sent_len", F.col("n_words") / F.greatest(F.col("n_sentences"), F.lit(1))))

# keep documents that are long enough to be worth summarising
df = df.filter((F.col("n_sentences") >= 5) & (F.col("n_words") >= 80))

df.select("doc_id", "category", "title", "text", "sentences", "tokens",
          "n_chars", "n_words", "n_sentences", "n_tokens_clean", "avg_sent_len") \
  .repartition(8).write.mode("overwrite").parquet(CLEAN_PQ)

out = spark.read.parquet(CLEAN_PQ)
print(f"Documents kept after cleaning/filtering : {out.count():,}")
out.select("doc_id", "category", "n_words", "n_sentences", "n_tokens_clean").show(8)
out.agg(F.round(F.mean("n_words"), 1).alias("mean_words"),
        F.expr("percentile_approx(n_words, 0.5)").alias("median_words"),
        F.max("n_words").alias("max_words"),
        F.round(F.mean("n_sentences"), 1).alias("mean_sentences")).show()
spark.stop()
