"""Assemble a self-describing run directory from cached stage artifacts."""
import joblib
import numpy as np

from nlp import common as C
from nlp import concordance as CO
from nlp import ingest as I
from nlp import segment as S
from nlp import similarities as SI
from nlp import stability as ST
from nlp import topics_nmf as T


def stability_cache_path(ctx, ing, sm):
    return ctx.manifests_dir / ("stability_%s_%s.json" % (ing["corpus_hash"][:8], sm["segmentation_version"]))


def tid(k):
    return "T%02d" % k


def load_optional(path):
    """Unavailable state instead of fabricated values when an artifact is missing."""
    if path.exists():
        return {"status": "available", "data": C.read_json(path)}
    return {"status": "unavailable", "data": None}


def labels_lookup(ctx):
    p = ctx.results_dir / "topic_labels.csv"
    if not p.exists():
        return {}
    return {(r["model_id"], r["topic_id"]): r for r in C.read_csv(p)}


def run_export(ctx):
    passages, sm = S.load_passages(ctx)
    ing = C.read_json(ctx.ingest_manifest_path())
    reg = ctx.load_models_registry()
    chash, segver = ing["corpus_hash"], sm["segmentation_version"]

    models = []
    for mid, r in reg.items():
        if r["kind"] == "refit" and r["corpus_hash"] == chash and r["segmentation_version"] == segver:
            models.append((mid, r))
        elif r["kind"] == "frozen" and r.get("classified_corpus_hash") == chash \
                and r.get("classified_segmentation_version") == segver:
            models.append((mid, r))
    models.sort(key=lambda m: (m[1]["K"], m[1]["kind"] != "refit", m[0]))
    if not models:
        raise C.StageError("export", "no NMF models for the current corpus; run fit-nmf first")
    sims = [(m, r) for m, r in reg.items() if r["kind"] == "similarity" and r["corpus_hash"] == chash
            and r["segmentation_version"] == segver]

    run_id = "%s-%s-%s" % (chash[:8], segver, C.cfg_hash(
        {"models": [m for m, _ in models], "sim": [m for m, _ in sims],
         "n": ctx.cfg["nmf"], "s": ctx.cfg["similarity"]}, 6))
    out = ctx.results_dir / run_id
    if ctx.dry_run:
        ctx.log.info("export (dry run): would write run %s with %d models to %s", run_id, len(models), out)
        return run_id
    out.mkdir(parents=True, exist_ok=True)
    docs = [d for d in ing["documents"]]
    docs_by_id = {d["video_id"]: d for d in docs}
    elig = T.eligible(passages)
    elig_words, excl_words, pcount = {}, {}, {}
    for p in passages:
        pcount[p["video_id"]] = pcount.get(p["video_id"], 0) + 1
        tgt = excl_words if p["excluded_from_content_analysis"] else elig_words
        tgt[p["video_id"]] = tgt.get(p["video_id"], 0) + p["word_count"]
    labels = labels_lookup(ctx)
    files = {}

    def emit(name, fields, rows):
        C.write_csv(out / name, fields, rows)
        files[name] = str((out / name).relative_to(ctx.root))

    # documents.csv
    dfields = ["run_id"] + I.DOC_FIELDS + ["passage_count", "eligible_words", "excluded_words"]
    emit("documents.csv", dfields, [dict(run_id=run_id, **d, passage_count=pcount.get(d["video_id"], 0),
                                          eligible_words=elig_words.get(d["video_id"], 0),
                                          excluded_words=excl_words.get(d["video_id"], 0)) for d in docs])
    # passages.csv
    emit("passages.csv", ["run_id"] + S.PASSAGE_FIELDS,
         [dict(run_id=run_id, **{k: p[k] for k in S.PASSAGE_FIELDS}) for p in passages])

    pt_rows, dt_rows, tp_rows, tt_rows, conc_rows = [], [], [], [], []
    model_reports = []
    pid_by_idx = [p["passage_id"] for p in elig]
    vids_by_idx = [p["video_id"] for p in elig]
    meta = [(p["video_id"], p["word_count"]) for p in elig]
    ncfg = ctx.cfg["nmf"]
    for mid, r in models:
        W = np.load(ctx.root / r["weights_file"])
        saved_ids = C.read_json(ctx.features_dir / (mid + ".passage_ids.json"))
        if saved_ids != pid_by_idx:
            raise C.StageError("export", "weights for %s do not match current passages; refit or reclassify" % mid)
        parent = mid if r["kind"] == "refit" else r["parent_model_id"]
        mdl = joblib.load(ctx.root / reg[parent]["model_file"])
        K = r["K"]
        H = mdl["nmf"].components_
        tt = T.top_terms(H, mdl["feature_names"], ncfg["top_terms"])
        sums = W.sum(axis=1)
        share = np.divide(W, sums[:, None], out=np.zeros_like(W), where=sums[:, None] > 0)
        dom = W.argmax(axis=1)
        agg = T.aggregate_documents(W, meta)
        vids_sorted = [d["video_id"] for d in docs if d["video_id"] in agg]
        for i, pid in enumerate(pid_by_idx):
            for k in range(K):
                if W[i, k] > 0:
                    pt_rows.append({"run_id": run_id, "model_id": mid, "passage_id": pid, "topic_id": tid(k),
                                    "weight": "%.6f" % W[i, k], "share": "%.6f" % share[i, k]})
        for v in vids_sorted:
            a = agg[v]
            status = "ok" if a["assigned_words"] > 0 else "unassigned"
            for k in range(K):
                dt_rows.append({"run_id": run_id, "model_id": mid, "video_id": v, "topic_id": tid(k),
                                "weight": "%.6f" % a["weights"][k], "assigned_words": a["assigned_words"],
                                "eligible_words": a["total_words"], "status": status})
        prominent = np.array([[agg[v]["weights"][k] >= ncfg["prominent_min"] for k in range(K)]
                              for v in vids_sorted]).sum(axis=0) if vids_sorted else np.zeros(K)
        for k in range(K):
            lab = labels.get((mid, tid(k)), {})
            tp_rows.append({"run_id": run_id, "model_id": mid, "model_kind": r["kind"], "parent_model_id": parent if r["kind"] == "frozen" else "",
                            "n_topics": K, "topic_id": tid(k), "label": lab.get("label", ""),
                            "label_source": "investigator" if lab.get("label") else "",
                            "n_passages_dominant": int((dom == k).sum()),
                            "n_lectures_prominent": int(prominent[k]),
                            "top_terms": "; ".join(t for t, _ in tt[k][:8])})
            for rank, (term, w) in enumerate(tt[k], 1):
                tt_rows.append({"run_id": run_id, "model_id": mid, "topic_id": tid(k), "rank": rank,
                                "term": term, "weight": "%.6f" % w})
            for rank, i in enumerate(T.representative_indices(W, vids_by_idx, k, ncfg["representative_passages"],
                                                              ncfg["max_per_lecture"]), 1):
                p = elig[i]
                d = docs_by_id[p["video_id"]]
                conc_rows.append({"run_id": run_id, "model_id": mid, "topic_id": tid(k), "rank": rank,
                                  "passage_id": p["passage_id"], "video_id": p["video_id"], "title": d["title"],
                                  "upload_date": d["upload_date"], "start_seconds": p["start_seconds"],
                                  "weight": "%.6f" % W[i, k],
                                  "youtube_url": C.youtube_url(p["video_id"], p["start_seconds"]),
                                  "text": p["text"]})
        model_reports.append({
            "model_id": mid, "kind": r["kind"], "n_topics": K, "converged": r.get("converged"),
            "n_iter": r.get("n_iter"), "reconstruction_err": r.get("reconstruction_err"),
            "unassigned_passages": int((sums <= 0).sum()),
            "documents_unassigned": [v for v in vids_sorted if agg[v]["assigned_words"] == 0],
            "topics_with_no_prominent_lecture": [tid(k) for k in range(K) if prominent[k] == 0],
            "topics_with_no_dominant_passage": [tid(k) for k in range(K) if (dom == k).sum() == 0]})

    emit("topics.csv", ["run_id", "model_id", "model_kind", "parent_model_id", "n_topics", "topic_id", "label",
                        "label_source", "n_passages_dominant", "n_lectures_prominent", "top_terms"], tp_rows)
    emit("topic_terms.csv", ["run_id", "model_id", "topic_id", "rank", "term", "weight"], tt_rows)
    emit("passage_topics.csv", ["run_id", "model_id", "passage_id", "topic_id", "weight", "share"], pt_rows)
    emit("document_topics.csv", ["run_id", "model_id", "video_id", "topic_id", "weight", "assigned_words",
                                 "eligible_words", "status"], dt_rows)
    emit("topic_concordances.csv", CO.CONC_FIELDS, conc_rows)

    # similarity (optional)
    sim_status = "unavailable"
    sim_rows, term_rows = [], []
    if sims:
        smid = sims[0][0]
        vids = C.read_json(ctx.features_dir / (smid + ".vids.json"))
        Sm = np.load(ctx.features_dir / (smid + ".S.npy"))
        sim_rows = SI.pair_rows(vids, Sm, "tfidf_cosine", smid, run_id, ctx.cfg["similarity"]["nn_limit"])
        term_rows = [dict(run_id=run_id, **t) for t in C.read_json(ctx.features_dir / (smid + ".terms.json"))]
        sim_status = "available"
    emit("document_similarity.csv", SI.PAIR_FIELDS, sim_rows)
    emit("document_terms.csv", SI.TERM_FIELDS, term_rows)

    # stability (optional)
    stab = load_optional(stability_cache_path(ctx, ing, sm))
    stab_rows = [dict(run_id=run_id, **r) for r in (stab["data"] or [])]
    emit("stability.csv", ST.STAB_FIELDS, stab_rows)

    manifest = {
        "run_id": run_id, "pipeline_version": C.PIPELINE_VERSION, "executed_at": C.now_iso(),
        "corpus": {"corpus_hash": chash, "scope": ing["corpus_scope"],
                   "included_documents": [{"video_id": d["video_id"], "normalized_hash": d["normalized_hash"]}
                                          for d in docs if d["processing_status"] == "ok"]},
        "preprocessing": {"normalize": ing["normalize_config"], "segmentation_version": segver,
                          "segment": sm["segment_config"], "boilerplate": sm["boilerplate_config"]},
        "software": C.package_versions(),
        "models": [{"model_id": m, **{k: v for k, v in r.items() if k not in ("model_file", "weights_file")}}
                   for m, r in models],
        "similarity_models": [{"model_id": m, **r} for m, r in sims],
        "nmf_config": ncfg,
        "representative_passage_ranking": "raw NMF loading W[p,k] descending, at most %d per lecture, "
                                          "top %d per topic" % (ncfg["max_per_lecture"], ncfg["representative_passages"]),
        "outputs": files,
    }
    C.write_json(out / "report.json", {
        "run_id": run_id,
        "counts": {"documents_ok": sum(d["processing_status"] == "ok" for d in docs),
                   "documents_error": sum(d["processing_status"] != "ok" for d in docs),
                   "passages": len(passages), "passages_eligible": len(elig),
                   "passages_excluded": len(passages) - len(elig),
                   "words_eligible": sum(p["word_count"] for p in elig)},
        "models": model_reports,
        "components": {"similarity_tfidf": sim_status, "stability": stab["status"],
                       "embedding_similarity": "unavailable", "bertopic": "unavailable"},
        "diagnostics": ing["diagnostics"],
        "interpretation_constraints": [
            "Topic weights are model-derived descriptive weights, not probabilities of endorsement.",
            "Lexical, topical and semantic similarity do not establish argumentative agreement or influence.",
            "Topic labels are investigator interpretations; model topic identifiers are neutral.",
            "Passages excluded as boilerplate are listed in passages.csv with reasons and are not modeled."],
    })
    files["report.json"] = str((out / "report.json").relative_to(ctx.root))
    # manifest last: its presence marks a completed run
    C.write_json(out / "manifest.json", manifest)
    C.atomic_write_text(ctx.results_dir / "latest_run.txt", run_id + "\n")
    ctx.log.info("export: run %s written to %s", run_id, out)
    return run_id
