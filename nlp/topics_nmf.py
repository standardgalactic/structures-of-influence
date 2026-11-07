"""Passage-level NMF topic discovery, frozen classification, document aggregation."""
import warnings

import joblib
import numpy as np
from sklearn.decomposition import NMF

from nlp import common as C
from nlp import features as F
from nlp import segment as S


def model_id_for(K, corpus_hash, segver, nmf_cfg):
    base = {k: v for k, v in nmf_cfg.items() if k != "topics"}
    return "nmf%02d-%s-%s-%s" % (K, corpus_hash[:8], segver, C.cfg_hash(base, 6))


def eligible(passages):
    return [p for p in passages if not p["excluded_from_content_analysis"]]


def fit_all(ctx, topics=None, force=False):
    passages, sm = S.load_passages(ctx)
    ing = C.read_json(ctx.ingest_manifest_path())
    corpus_hash, segver = ing["corpus_hash"], sm["segmentation_version"]
    elig = eligible(passages)
    ncfg = ctx.cfg["nmf"]
    topics = topics or ncfg["topics"]
    if ctx.dry_run:
        for K in topics:
            ctx.log.info("fit-nmf (dry run): would fit %s on %d passages",
                         model_id_for(K, corpus_hash, segver, ncfg), len(elig))
        return []
    if not elig:
        raise C.StageError("fit-nmf", "no eligible passages (all excluded or corpus empty)")
    ctx.ensure_dirs()
    fid, X, vec = F.fit_or_load(ctx, corpus_hash, segver, elig, force=force)
    terms = np.array(vec.get_feature_names_out())
    reg = ctx.load_models_registry()
    out = []
    for K in topics:
        mid = model_id_for(K, corpus_hash, segver, ncfg)
        mpath = ctx.models_dir / (mid + ".joblib")
        wpath = ctx.features_dir / (mid + ".W.npy")
        if not force and mpath.exists() and wpath.exists() and mid in reg:
            ctx.log.info("fit-nmf: %s already fitted, reusing", mid)
            out.append(mid)
            continue
        if K >= min(X.shape):
            raise C.StageError("fit-nmf", "K=%d is not smaller than the data dimensions %s" % (K, X.shape))
        ctx.log.info("fit-nmf: fitting K=%d on %s", K, X.shape)
        nmf = NMF(n_components=K, init=ncfg["init"], random_state=ncfg["seed"], max_iter=ncfg["max_iter"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = nmf.fit_transform(X)
        joblib.dump({"model_id": mid, "vectorizer": vec, "nmf": nmf, "feature_names": terms,
                     "config": ncfg, "K": K}, mpath)
        np.save(wpath, W.astype(np.float64))
        C.write_json(ctx.features_dir / (mid + ".passage_ids.json"), [p["passage_id"] for p in elig])
        reg[mid] = {"kind": "refit", "K": K, "corpus_hash": corpus_hash, "segmentation_version": segver,
                    "features_id": fid, "created": C.now_iso(), "seed": ncfg["seed"],
                    "n_iter": int(nmf.n_iter_), "max_iter": ncfg["max_iter"],
                    "converged": bool(nmf.n_iter_ < ncfg["max_iter"]),
                    "reconstruction_err": float(nmf.reconstruction_err_),
                    "model_file": str(mpath.relative_to(ctx.root)),
                    "weights_file": str(wpath.relative_to(ctx.root)),
                    "n_passages": int(X.shape[0]), "n_terms": int(X.shape[1])}
        ctx.save_models_registry(reg)
        out.append(mid)
    return out


def classify_frozen(ctx, parent_model_id):
    """Apply an existing fitted vectorizer + NMF to the current passages without refitting."""
    reg = ctx.load_models_registry()
    if parent_model_id not in reg:
        raise C.StageError("classify-frozen", "unknown model id %s" % parent_model_id)
    passages, sm = S.load_passages(ctx)
    ing = C.read_json(ctx.ingest_manifest_path())
    elig = eligible(passages)
    parent = joblib.load(ctx.root / reg[parent_model_id]["model_file"])
    mid = "%s.frozen-%s" % (parent_model_id, ing["corpus_hash"][:8])
    if ctx.dry_run:
        ctx.log.info("classify-frozen (dry run): %s -> %s on %d passages", parent_model_id, mid, len(elig))
        return mid
    X = parent["vectorizer"].transform([p["text"] for p in elig])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = parent["nmf"].transform(X)
    wpath = ctx.features_dir / (mid + ".W.npy")
    np.save(wpath, W)
    C.write_json(ctx.features_dir / (mid + ".passage_ids.json"), [p["passage_id"] for p in elig])
    rec = dict(reg[parent_model_id])
    rec.update({"kind": "frozen", "parent_model_id": parent_model_id,
                "classified_corpus_hash": ing["corpus_hash"],
                "classified_segmentation_version": sm["segmentation_version"],
                "weights_file": str(wpath.relative_to(ctx.root)), "created": C.now_iso(),
                "n_passages": int(X.shape[0])})
    reg[mid] = rec
    ctx.save_models_registry(reg)
    ctx.log.info("classify-frozen: %s written (%d passages)", mid, len(elig))
    return mid


def aggregate_documents(W, meta):
    """Word-count-weighted lecture-level topic proportions.

    W: (n_passages, K) non-negative weights. meta: list of (video_id, word_count).
    For lecture d and topic k: sum_p n_p * Wt[p,k] / sum_p n_p, where Wt is W
    row-normalized and the sums run over assigned passages. A passage whose
    weights sum to zero is unassigned and leaves the denominator.
    """
    W = np.asarray(W, dtype=float)
    sums = W.sum(axis=1)
    out = {}
    K = W.shape[1]
    for (vid, n), row, s in zip(meta, W, sums):
        rec = out.setdefault(vid, {"num": np.zeros(K), "assigned_words": 0, "total_words": 0,
                                   "unassigned_passages": 0})
        rec["total_words"] += n
        if s > 0:
            rec["num"] += n * (row / s)
            rec["assigned_words"] += n
        else:
            rec["unassigned_passages"] += 1
    result = {}
    for vid, rec in out.items():
        w = rec["num"] / rec["assigned_words"] if rec["assigned_words"] > 0 else np.zeros(K)
        result[vid] = {"weights": w, "assigned_words": rec["assigned_words"],
                       "total_words": rec["total_words"],
                       "unassigned_passages": rec["unassigned_passages"]}
    return result


def representative_indices(W, video_ids, k, per_topic, max_per_lecture):
    """Top passages for topic k ranked by raw NMF loading, at most max_per_lecture per lecture."""
    order = np.argsort(-W[:, k], kind="stable")
    chosen, per = [], {}
    for i in order:
        if W[i, k] <= 0:
            break
        v = video_ids[i]
        if per.get(v, 0) >= max_per_lecture:
            continue
        per[v] = per.get(v, 0) + 1
        chosen.append(int(i))
        if len(chosen) >= per_topic:
            break
    return chosen


def top_terms(H, terms, n):
    out = []
    for k in range(H.shape[0]):
        idx = np.argsort(-H[k], kind="stable")[:n]
        out.append([(str(terms[j]), float(H[k, j])) for j in idx if H[k, j] > 0])
    return out
