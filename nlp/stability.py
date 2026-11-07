"""Categorization stability: repeated fits, alternative topic counts, alternative representation."""
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from nlp import common as C
from nlp import features as F
from nlp import segment as S
from nlp import topics_nmf as T

STAB_FIELDS = ["run_id", "comparison", "model_a", "model_b", "metric", "value", "note"]


def _unit(H):
    n = np.linalg.norm(H, axis=1, keepdims=True)
    n[n == 0] = 1
    return H / n


def match_topics(Ha, Hb):
    """One-to-one matching of topics by cosine similarity of term weights (Hungarian).

    Topic identifiers are permutation-invariant, so matching is required before
    comparing term representations. Requires equal K.
    """
    sim = _unit(Ha) @ _unit(Hb).T
    r, c = linear_sum_assignment(-sim)
    return sim[r, c], c


def coverage(Ha, Hb):
    """For unequal K: mean over topics of A of the best cosine to any topic of B."""
    sim = _unit(Ha) @ _unit(Hb).T
    return float(sim.max(axis=1).mean())


def co_assignment(da, db):
    """Mean over topics of A of the share of its dominant passages that fall in one B topic."""
    vals = []
    for k in np.unique(da):
        members = db[da == k]
        vals.append(np.bincount(members).max() / len(members))
    return float(np.mean(vals))


def run_stability(ctx, run_id_hint=""):
    import joblib
    passages, sm = S.load_passages(ctx)
    ing = C.read_json(ctx.ingest_manifest_path())
    reg = ctx.load_models_registry()
    refit = {m: r for m, r in reg.items() if r["kind"] == "refit"
             and r["corpus_hash"] == ing["corpus_hash"]
             and r["segmentation_version"] == sm["segmentation_version"]}
    if not refit:
        raise C.StageError("stability", "no fitted NMF models for the current corpus; run fit-nmf first")
    if ctx.dry_run:
        ctx.log.info("stability (dry run): %d models", len(refit))
        return []
    elig = T.eligible(passages)
    rows = []
    models = {}
    for mid, r in sorted(refit.items(), key=lambda kv: kv[1]["K"]):
        mdl = joblib.load(ctx.root / r["model_file"])
        W = np.load(ctx.root / r["weights_file"])
        models[mid] = (r["K"], mdl, W)
    _, X, _ = F.fit_or_load(ctx, ing["corpus_hash"], sm["segmentation_version"], elig)
    ncfg = ctx.cfg["nmf"]
    n_seeds = ctx.cfg["stability"]["seeds"]
    for mid, (K, mdl, W) in models.items():
        base_dom = W.argmax(axis=1)
        for s in range(1, n_seeds + 1):
            ctx.log.info("stability: %s repeat fit seed=%d", mid, ncfg["seed"] + s)
            nm = NMF(n_components=K, init="random", random_state=ncfg["seed"] + s, max_iter=ncfg["max_iter"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W2 = nm.fit_transform(X)
            sims, _ = match_topics(mdl["nmf"].components_, nm.components_)
            tag = "random_init_seed_%d" % (ncfg["seed"] + s)
            dom2 = W2.argmax(axis=1)
            for metric, val in (("mean_matched_topic_cosine", float(sims.mean())),
                                ("min_matched_topic_cosine", float(sims.min())),
                                ("adjusted_rand_index", adjusted_rand_score(base_dom, dom2)),
                                ("adjusted_mutual_info", adjusted_mutual_info_score(base_dom, dom2))):
                rows.append({"comparison": "repeat_fit", "model_a": mid, "model_b": tag, "metric": metric,
                             "value": "%.6f" % val,
                             "note": "NMF nndsvd primary fit vs random-init refit; passage dominant-topic partitions"})
    ids = list(models)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            Ka, ma, Wa = models[ids[i]]
            Kb, mb, Wb = models[ids[j]]
            da, db = Wa.argmax(axis=1), Wb.argmax(axis=1)
            note = "different K (%d vs %d): coverage and co-assignment, not one-to-one matching" % (Ka, Kb)
            for metric, val in (("coverage_a_in_b", coverage(ma["nmf"].components_, mb["nmf"].components_)),
                                ("coverage_b_in_a", coverage(mb["nmf"].components_, ma["nmf"].components_)),
                                ("co_assignment_a_in_b", co_assignment(da, db)),
                                ("co_assignment_b_in_a", co_assignment(db, da)),
                                ("adjusted_rand_index", adjusted_rand_score(da, db)),
                                ("adjusted_mutual_info", adjusted_mutual_info_score(da, db))):
                rows.append({"comparison": "topic_count", "model_a": ids[i], "model_b": ids[j],
                             "metric": metric, "value": "%.6f" % val, "note": note})
    # Alternative representation: document-level TF-IDF similarity vs NMF document-topic profiles.
    simmods = [m for m, r in reg.items() if r["kind"] == "similarity" and r["corpus_hash"] == ing["corpus_hash"]]
    if simmods:
        smid = simmods[0]
        vids = C.read_json(ctx.features_dir / (smid + ".vids.json"))
        Sm = np.load(ctx.features_dir / (smid + ".S.npy"))
        meta = [(p["video_id"], p["word_count"]) for p in elig]
        for mid, (K, mdl, W) in models.items():
            agg = T.aggregate_documents(W, meta)
            P = np.array([agg[v]["weights"] for v in vids])
            Pn = _unit(P)
            Sp = np.clip(Pn @ Pn.T, 0, 1)
            iu = np.triu_indices(len(vids), 1)
            if len(iu[0]) > 2:
                from scipy.stats import spearmanr
                rho = spearmanr(Sm[iu], Sp[iu]).correlation
                rows.append({"comparison": "alternative_representation", "model_a": mid, "model_b": smid,
                             "metric": "spearman_pairwise_similarity", "value": "%.6f" % rho,
                             "note": "lecture pair similarity from NMF topic profiles vs lecture TF-IDF cosine"})
    return rows
