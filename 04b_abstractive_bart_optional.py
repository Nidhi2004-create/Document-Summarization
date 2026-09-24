"""
OPTIONAL abstractive summariser (Hugging Face BART/DistilBART) - run on your own machine / Colab.
    pip install transformers torch
It downloads a pre-trained model (~1.2 GB) so it needs internet access to huggingface.co.
"""
import pandas as pd
from transformers import pipeline
from common import RAW_CSV

summariser = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
df = pd.read_csv(RAW_CSV).query("split == 'test'").head(5)
for _, r in df.iterrows():
    out = summariser(r.text[:3000], max_length=90, min_length=30, do_sample=False)[0]["summary_text"]
    print(f"\n### {r.title}\n{out}")
