"""Light-weight extractive summariser (same idea as Spark model M2) used as the chatbot's offline engine."""
import re, math
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP


def split_sentences(text):
    parts = re.split(r"(?<=[a-z0-9\"\)][.!?])\s+(?=[A-Z\"\(])", " ".join(text.split()))
    return [p.strip() for p in parts if len(p.split()) >= 4]


def extractive_summary(text, k=3):
    sents = split_sentences(text)
    if len(sents) <= k:
        return sents
    toks = [[w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP] for s in sents]
    tf = Counter(w for t in toks for w in t)
    dfc = Counter(w for t in toks for w in set(t))
    n = len(sents)
    scores = []
    for i, t in enumerate(toks):
        s = sum(tf[w] * (math.log((n + 1) / (dfc[w] + 1)) + 1) for w in t) / math.sqrt(max(len(t), 1))
        scores.append(s * (1 + 1 / (1 + i)))
    top = sorted(sorted(range(n), key=lambda i: -scores[i])[:k])
    return [sents[i] for i in top]


def clean_wire_text(text):
    """Remove Reuters <TICKER.N> tags and unwrap <Company Name> markers."""
    text = re.sub(r"\s*<[A-Z0-9.\-]{1,10}>", "", text)
    return re.sub(r"<([^<>]+)>", r"\1", text)
