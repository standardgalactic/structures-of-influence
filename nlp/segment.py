"""Passage segmentation and auditable boilerplate flagging."""
import hashlib
import math
import re

from nlp import common as C
from nlp import ingest as I
from nlp import normalize as N

SENT_END = re.compile(r"[.?!]['\")\]]*$")

PASSAGE_FIELDS = ["passage_id", "video_id", "passage_index", "start_seconds", "end_seconds",
                  "word_count", "text", "excluded_from_content_analysis", "exclusion_reason",
                  "boilerplate_fraction", "segmentation_version"]


def segmentation_version(seg_cfg, boiler_cfg):
    return "%s-%s" % (seg_cfg["version"], C.cfg_hash({"s": seg_cfg, "b": boiler_cfg}, 6))


def segment_tokens(tokens, cfg):
    """Return contiguous (start, end) index ranges of roughly equal size.

    Boundaries move to the nearest sentence end within a slack window when the
    transcript has punctuation; otherwise they fall at the even split points.
    min_words is a soft limit (a short lecture yields one short passage).
    """
    n = len(tokens)
    if n == 0:
        return []
    target, lo, hi, slack = cfg["target_words"], cfg["min_words"], cfg["max_words"], cfg["boundary_slack"]
    count = max(1, int(round(n / target)))
    while math.ceil(n / count) > hi:
        count += 1
    if count > 1 and n / count < lo and math.ceil(n / (count - 1)) <= hi:
        count -= 1
    bounds, prev = [0], 0
    for i in range(1, count):
        ideal = int(round(i * n / count))
        best = ideal
        for off in range(0, slack + 1):
            hit = None
            for cand in (ideal - off, ideal + off):
                if prev + lo <= cand <= min(prev + hi, n - 1) and SENT_END.search(tokens[cand - 1]):
                    hit = cand
                    break
            if hit is not None:
                best = hit
                break
        best = max(best, prev + 1)
        bounds.append(best)
        prev = best
    bounds.append(n)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def _norm_words(tokens):
    return [re.sub(r"[^\w']", "", t.lower()) for t in tokens]


def _shingles(words, k):
    for i in range(0, max(0, len(words) - k + 1)):
        yield hash(" ".join(words[i:i + k]))


def flag_boilerplate(passages, cfg):
    """Annotate passages in place. Whole-passage exclusion only, with the reason recorded.

    Cross-document near-duplicate detection: a passage is flagged when at least
    min_fraction of its word shingles occur in at least min_docs distinct lectures.
    Passages arrive grouped by video, so a per-shingle (count, last_video) pair
    gives an exact distinct-lecture count.
    """
    for p in passages:
        p["excluded_from_content_analysis"] = False
        p["exclusion_reason"] = ""
        p["boilerplate_fraction"] = 0.0
    if not cfg["enabled"]:
        return
    k, min_docs, min_frac = cfg["shingle_words"], cfg["min_docs"], cfg["min_fraction"]
    seen = {}
    shingles = []
    for p in passages:
        sh = list(_shingles(_norm_words(p["_tokens"]), k))
        shingles.append(sh)
        for h in set(sh):
            rec = seen.get(h)
            if rec is None:
                seen[h] = [1, p["video_id"]]
            elif rec[1] != p["video_id"]:
                rec[0] += 1
                rec[1] = p["video_id"]
    for p, sh in zip(passages, shingles):
        if sh:
            shared = sum(1 for h in sh if seen[h][0] >= min_docs)
            p["boilerplate_fraction"] = round(shared / len(sh), 4)
        reasons = []
        if p["boilerplate_fraction"] >= min_frac:
            reasons.append("cross_document_near_duplicate(fraction=%.2f,min_docs=%d)"
                           % (p["boilerplate_fraction"], min_docs))
        for rule in cfg.get("lexical_rules", []):
            n = len(re.findall(rule["pattern"], p["text"], flags=re.I))
            if n >= rule.get("min_matches", 3):
                reasons.append("lexical_rule:%s(matches=%d)" % (rule["name"], n))
        if reasons:
            p["excluded_from_content_analysis"] = True
            p["exclusion_reason"] = ";".join(reasons)


def passages_path(ctx, corpus_hash, segver):
    return ctx.passages_dir / ("passages_%s_%s.csv" % (corpus_hash[:8], segver))


def run_segment(ctx):
    mpath = ctx.ingest_manifest_path()
    if not mpath.exists():
        raise C.StageError("segment", "no ingest manifest found; run the ingest stage first")
    ing = C.read_json(mpath)
    seg_cfg, b_cfg = ctx.cfg["segment"], ctx.cfg["boilerplate"]
    segver = segmentation_version(seg_cfg, b_cfg)
    out = passages_path(ctx, ing["corpus_hash"], segver)
    docs = [d for d in ing["documents"] if d["processing_status"] == "ok"]
    if ctx.dry_run:
        ctx.log.info("segment (dry run): %d documents -> %s", len(docs), out)
        return {"passages": [], "segmentation_version": segver}
    passages = []
    for i, d in enumerate(docs, 1):
        try:
            payload = I.load_doc_cache(ctx, d["video_id"])
        except Exception as exc:
            raise C.StageError("segment", "cannot read cached transcript: %s" % exc, d["source_path"])
        tokens = payload["tokens"]
        starts, ends = N.token_times(payload["runs"])
        for idx, (a, b) in enumerate(segment_tokens(tokens, seg_cfg)):
            pid = "%s_p%03d_%s" % (d["video_id"], idx,
                                   C.sha256_text("%s|%s|%d" % (d["normalized_hash"], segver, idx))[:6])
            passages.append({
                "passage_id": pid, "video_id": d["video_id"], "passage_index": idx,
                "start_seconds": round(starts[a], 3), "end_seconds": round(ends[b - 1], 3),
                "word_count": b - a, "text": " ".join(tokens[a:b]), "_tokens": tokens[a:b],
                "segmentation_version": segver,
            })
        if i % 25 == 0 or i == len(docs):
            ctx.log.info("segment: %d/%d documents", i, len(docs))
    flag_boilerplate(passages, b_cfg)
    rows = [{k: p[k] for k in PASSAGE_FIELDS} for p in passages]
    C.write_csv(out, PASSAGE_FIELDS, rows)
    excluded = sum(1 for p in passages if p["excluded_from_content_analysis"])
    manifest = {"stage": "segment", "pipeline_version": C.PIPELINE_VERSION, "executed_at": C.now_iso(),
                "corpus_hash": ing["corpus_hash"], "segmentation_version": segver,
                "segment_config": seg_cfg, "boilerplate_config": b_cfg,
                "passages_file": str(out.relative_to(ctx.root)),
                "n_passages": len(passages), "n_excluded": excluded,
                "excluded_words": sum(p["word_count"] for p in passages if p["excluded_from_content_analysis"]),
                "limitation": "Exclusion is applied to whole passages; short boilerplate embedded in a "
                              "longer passage remains. boilerplate_fraction is recorded for audit."}
    C.write_json(ctx.segment_manifest_path(), manifest)
    ctx.log.info("segment: %d passages (%d excluded) -> %s", len(passages), excluded, out)
    return {"passages": passages, "segmentation_version": segver, "path": out}


def load_passages(ctx):
    """Read passages for the current ingest + segment manifests."""
    if not ctx.segment_manifest_path().exists():
        raise C.StageError("segment", "no segment manifest found; run the segment stage first")
    sm = C.read_json(ctx.segment_manifest_path())
    path = ctx.root / sm["passages_file"]
    rows = C.read_csv(path)
    for r in rows:
        r["passage_index"] = int(r["passage_index"])
        r["start_seconds"] = float(r["start_seconds"])
        r["end_seconds"] = float(r["end_seconds"])
        r["word_count"] = int(r["word_count"])
        r["excluded_from_content_analysis"] = r["excluded_from_content_analysis"] == "True"
        r["boilerplate_fraction"] = float(r["boilerplate_fraction"] or 0)
    return rows, sm
