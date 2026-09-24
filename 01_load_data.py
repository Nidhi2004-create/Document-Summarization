"""
Task 1 - Store / load the document dataset (AWS S3 or local storage).

Dataset : Reuters-21578 news corpus (10,788 articles) shipped with NLTK.
Output  : data/reuters_raw.csv  (doc_id, split, category, title, text)
S3      : if the env var S3_BUCKET is set, the CSV is also uploaded to S3
          and re-loaded from S3 (uses boto3 + your AWS credentials).
"""
import os, sys, html
import pandas as pd
import nltk
from common import RAW_CSV, S3_BUCKET, S3_KEY

nltk.download("reuters", quiet=True)
from nltk.corpus import reuters


def build_dataframe():
    rows = []
    for fid in reuters.fileids():
        raw = reuters.raw(fid)
        title, _, body = raw.partition("\n")            # 1st line = headline
        body = " ".join(html.unescape(body).split())    # decode &lt; &amp; ... and collapse whitespace
        cats = reuters.categories(fid)
        rows.append(dict(doc_id=fid.replace("/", "_"),
                         split=fid.split("/")[0],
                         category=cats[0] if cats else "none",
                         title=html.unescape(title).strip().title(),
                         text=body))
    return pd.DataFrame(rows)


def upload_to_s3(path):
    import boto3
    boto3.client("s3").upload_file(path, S3_BUCKET, S3_KEY,
                                   ExtraArgs={"ServerSideEncryption": "aws:kms"})
    print(f"Uploaded to s3://{S3_BUCKET}/{S3_KEY} (SSE-KMS encrypted)")


def download_from_s3(path):
    import boto3
    boto3.client("s3").download_file(S3_BUCKET, S3_KEY, path)
    print(f"Downloaded s3://{S3_BUCKET}/{S3_KEY} -> {path}")


if __name__ == "__main__":
    df = build_dataframe()
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(df):,} documents to {RAW_CSV}")
    print(df[["doc_id", "category", "title"]].head(5).to_string(index=False))
    if S3_BUCKET:
        upload_to_s3(RAW_CSV)
        download_from_s3(RAW_CSV)
