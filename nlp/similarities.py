"""Document-level similarity (TF-IDF cosine). Embedding similarity is Phase IV and kept separate."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp import common as C
from nlp import segment as S
from nlp import topics_nmf as T

PAIR_FIELDS = ["run_id", "video_id_a", "video_id_b", "method", "similarity", "model_id"]
TERM_FIELDS = ["run_id", "model_id", "video_id", "rank", "term", "weight"]


def similarity_model_id(corpus_hash, segver, scfg):
    return "tfidfdoc-%s-%s-%s" % (corpus_hash[:8], segver, C.cfg_hash(scfg, 6))


def doc_matrix(doc_texts, scfg):
    vids = list(doc_texts)
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=scfg["min_df"],
                          max_df=scfg["max_df"], sublinear_tf=True, norm="l2",
                          stop_words=scfg["stop_words"])
    X = vec.fit_transform([doc_texts[v] for v in vids]).tocsr()
    return vids, vec, X


def cosine_matrix(X):
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 1.0)
    return np.clip(S, 0.0, 1.0)


def pair_rows(vids, S, method, model_id, run_id, nn_limit=0):
    rows = []
    n = len(vids)
    if nn_limit and nn_limit < n - 1:
        for i in range(n):
            order = [j for j in np.argsort(-S[i], kind="stable") if j != i][:nn_limit]
            for j in order:
                rows.append({"run_id": run_id, "video_id_a": vids[i], "video_id_b": vids[j],
                             "method": method, "similarity": "%.6f" % S[i, j], "model_id": model_id})
    else:
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({"run_id": run_id, "video_id_a": vids[i], "video_id_b": vids[j],
                             "method": method, "similarity": "%.6f" % S[i, j], "model_id": model_id})
    return rows


def run_similarities(ctx, method="tfidf"):
    if method != "tfidf":
        raise C.StageError("similarities", "method %r is not implemented (Phase IV adds embeddings)" % method)
    passages, sm = S.load_passages(ctx)
    ing = C.read_json(ctx.ingest_manifest_path())
    scfg = ctx.cfg["similarity"]
    mid = similarity_model_id(ing["corpus_hash"], sm["segmentation_version"], scfg)
    if ctx.dry_run:
        ctx.log.info("similarities (dry run): would compute %s", mid)
        return mid
    texts = {}
    for p in T.eligible(passages):
        texts.setdefault(p["video_id"], []).append(p["text"])
    texts = {v: " ".join(t) for v, t in texts.items()}
    if len(texts) < 2:
        raise C.StageError("similarities", "need at least two lectures with eligible text")
    vids, vec, X = doc_matrix(texts, scfg)
    Sm = cosine_matrix(X)
    terms = np.array(vec.get_feature_names_out())
    term_rows = []
    for i, v in enumerate(vids):
        row = X[i].toarray().ravel()
        for r, j in enumerate(np.argsort(-row, kind="stable")[: scfg["top_terms_per_document"]], 1):
            if row[j] > 0:
                term_rows.append({"model_id": mid, "video_id": v, "rank": r,
                                  "term": str(terms[j]), "weight": "%.6f" % row[j]})
    ctx.ensure_dirs()
    np.save(ctx.features_dir / (mid + ".S.npy"), Sm)
    C.write_json(ctx.features_dir / (mid + ".vids.json"), vids)
    C.write_json(ctx.features_dir / (mid + ".terms.json"), term_rows)
    reg = ctx.load_models_registry()
    reg[mid] = {"kind": "similarity", "method": "tfidf_cosine", "corpus_hash": ing["corpus_hash"],
                "segmentation_version": sm["segmentation_version"], "created": C.now_iso(),
                "config": scfg, "n_documents": len(vids), "n_terms": int(X.shape[1])}
    ctx.save_models_registry(reg)
    ctx.log.info("similarities: %d lectures, %d terms -> %s", len(vids), X.shape[1], mid)
    return mid
