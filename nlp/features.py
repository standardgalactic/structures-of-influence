"""TF-IDF feature extraction for passages (sparse, cached)."""
import warnings

import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp import common as C


def make_vectorizer(tfidf_cfg):
    kw = dict(tfidf_cfg)
    kw["ngram_range"] = tuple(kw["ngram_range"])
    return TfidfVectorizer(**kw)


def features_id(corpus_hash, segver, tfidf_cfg):
    return "tfidf-%s-%s-%s" % (corpus_hash[:8], segver, C.cfg_hash(tfidf_cfg, 6))


def fit_or_load(ctx, corpus_hash, segver, eligible, force=False):
    """Fit the passage-level vectorizer once per corpus/segmentation/config and cache it."""
    tcfg = ctx.cfg["nmf"]["tfidf"]
    fid = features_id(corpus_hash, segver, tcfg)
    xp = ctx.features_dir / (fid + ".npz")
    vp = ctx.features_dir / (fid + ".vectorizer.joblib")
    ip = ctx.features_dir / (fid + ".passage_ids.json")
    if not force and xp.exists() and vp.exists() and ip.exists():
        ids = C.read_json(ip)
        if ids == [p["passage_id"] for p in eligible]:
            ctx.log.info("features: reusing cached TF-IDF matrix %s", fid)
            return fid, sp.load_npz(xp).tocsr(), joblib.load(vp)
    vec = make_vectorizer(tcfg)
    texts = [p["text"] for p in eligible]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = vec.fit_transform(texts).tocsr()
    except ValueError as exc:
        raise C.StageError("features", "TF-IDF failed (corpus too small for min_df/max_df?): %s" % exc)
    ctx.log.info("features: %d passages x %d terms (nnz=%d)", X.shape[0], X.shape[1], X.nnz)
    tmp = xp.with_suffix(".tmp.npz")
    sp.save_npz(tmp, X)
    tmp.replace(xp)
    joblib.dump(vec, vp)
    C.write_json(ip, [p["passage_id"] for p in eligible])
    return fid, X, vec
