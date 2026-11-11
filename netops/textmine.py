"""Text mining on alarm lines: mask variable tokens, TF-IDF, KMeans.

Clusters are checked against the syslog mnemonic (%TAG-sev-NAME) that the
model never sees as a label: purity = share of lines whose cluster's
majority mnemonic matches their own."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

MASKS = [(re.compile(r"\b\d+\.\d+\.\d+\.\d+\b"), "<ip>"), (re.compile(r"ge-\d+/\d+/\d+"), "<iface>"),
         (re.compile(r"\b(core|agg|acc)-\d+\b"), "<router>"), (re.compile(r"\d+"), "<n>")]


def mask(msg: str) -> str:
    body = msg.split(": ", 1)[-1]                   # drop the mnemonic: clusters must come from the words
    for rx, tok in MASKS:
        body = rx.sub(tok, body)
    return body.lower()


def mnemonic(msg: str) -> str:
    return msg.split(":", 1)[0]


def cluster(alarms: list[dict], seed: int = 0):
    texts = [mask(a["msg"]) for a in alarms]
    vec = TfidfVectorizer(token_pattern=r"[a-z<>_\-]+", ngram_range=(1, 2), min_df=2)
    Xt = vec.fit_transform(texts)
    best = None
    for k in range(4, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xt)
        sil = silhouette_score(Xt, km.labels_, sample_size=min(800, Xt.shape[0]), random_state=seed)
        if best is None or sil > best[0]:
            best = (sil, k, km)
    sil, k, km = best
    terms = np.array(vec.get_feature_names_out())
    clusters = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        top = terms[np.argsort(-km.cluster_centers_[c])[:5]].tolist()
        mn = Counter(mnemonic(alarms[i]["msg"]) for i in idx)
        clusters.append({"id": c, "size": int(len(idx)), "top_terms": top, "examples": [alarms[i]["msg"] for i in idx[:3]],
                         "majority_mnemonic": mn.most_common(1)[0][0]})
    labels = km.labels_.tolist()
    purity = sum(mnemonic(a["msg"]) == clusters[labels[i]]["majority_mnemonic"] for i, a in enumerate(alarms)) / len(alarms)
    return labels, clusters, {"k": k, "silhouette": round(float(sil), 3), "purity": round(purity, 3), "lines": len(alarms)}


def main(alarms=None):
    if alarms is None:
        from .store import load_alarms, session
        truth = json.loads(Path("data/truth.json").read_text())
        alarms = load_alarms(session(), sorted(n["id"] for n in truth["topology"]["nodes"]))
    labels, clusters, stats = cluster(alarms)
    Path("results").mkdir(exist_ok=True)
    Path("results/alarm_clusters.json").write_text(json.dumps({"stats": stats, "clusters": clusters, "labels": labels}, indent=1))
    print(json.dumps(stats))
    for c in clusters:
        print(f"  c{c['id']} n={c['size']:<4} {c['majority_mnemonic']:<20} {' · '.join(c['top_terms'][:4])}")


if __name__ == "__main__":
    main()
