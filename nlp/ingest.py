"""Source discovery, video identity, deduplication of subtitle variants, caching."""
import gzip
import json
import re
import unicodedata

from nlp import common as C
from nlp import normalize as N

FILENAME_RE = re.compile(r"^(\d{8})_(.{11})_(.+)\.(en-orig|en)\.vtt$", re.S)

DOC_FIELDS = ["video_id", "upload_date", "title", "source_path", "language_variant",
              "source_hash", "normalized_hash", "word_count", "duration_seconds",
              "processing_status"]


def parse_filename(name):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    d, vid, title, variant = m.groups()
    date = "%s-%s-%s" % (d[:4], d[4:6], d[6:8])
    # yt-dlp substitutes full-width look-alikes for characters illegal in filenames
    title = unicodedata.normalize("NFKC", title)
    return {"upload_date": date, "video_id": vid, "title": title, "language_variant": variant}


def discover(ctx, diagnostics):
    """Return one chosen source per video id, in (date, video_id) order."""
    groups = {}
    for p in sorted((ctx.root / "transcripts").glob("*.vtt")):
        info = parse_filename(p.name)
        if info is None:
            diagnostics.append({"stage": "ingest", "source_path": p.name, "status": "skipped",
                                "message": "filename does not match YYYYMMDD_VIDEOID_TITLE.lang.vtt"})
            continue
        info["path"] = p
        groups.setdefault(info["video_id"], []).append(info)
    chosen = []
    for vid, items in groups.items():
        items.sort(key=lambda i: (i["language_variant"] != "en-orig", i["path"].name))
        chosen.append(items[0])
        for extra in items[1:]:
            diagnostics.append({"stage": "ingest", "source_path": extra["path"].name,
                                "status": "variant_skipped",
                                "message": "video %s already represented by %s" % (vid, items[0]["path"].name)})
    chosen.sort(key=lambda i: (i["upload_date"], i["video_id"]))
    if ctx.video_ids:
        chosen = [c for c in chosen if c["video_id"] in ctx.video_ids]
    if ctx.limit:
        chosen = chosen[: ctx.limit]
    return chosen


def doc_cache_path(ctx, video_id):
    return ctx.docs_dir / ("%s.json.gz" % video_id)


def load_doc_cache(ctx, video_id):
    with gzip.open(doc_cache_path(ctx, video_id), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _save_doc_cache(ctx, video_id, payload):
    C.atomic_write_bytes(doc_cache_path(ctx, video_id),
                         gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), 6))


def corpus_hash(documents):
    lines = sorted("%s:%s" % (d["video_id"], d["normalized_hash"]) for d in documents
                   if d["processing_status"] == "ok")
    return C.sha256_text("\n".join(lines))


def run_ingest(ctx):
    ctx.ensure_dirs() if not ctx.dry_run else None
    diagnostics = []
    chosen = discover(ctx, diagnostics)
    ctx.log.info("ingest: %d videos selected", len(chosen))
    if ctx.dry_run:
        for c in chosen:
            ctx.log.info("ingest (dry run): would process %s", c["path"].name)
        return {"documents": [], "diagnostics": diagnostics, "cache_hits": 0, "processed": 0}

    norm_cfg = ctx.cfg["normalize"]
    norm_id = C.cfg_hash(norm_cfg)
    documents, hits, processed = [], 0, 0
    for i, c in enumerate(chosen, 1):
        rel = c["path"].name
        rec = {"video_id": c["video_id"], "upload_date": c["upload_date"], "title": c["title"],
               "source_path": rel, "language_variant": c["language_variant"],
               "source_hash": "", "normalized_hash": "", "word_count": 0,
               "duration_seconds": "", "processing_status": "ok"}
        try:
            raw = c["path"].read_bytes()
            rec["source_hash"] = C.sha256_bytes(raw)
            cached = None
            if doc_cache_path(ctx, c["video_id"]).exists():
                try:
                    cc = load_doc_cache(ctx, c["video_id"])
                    if cc.get("source_hash") == rec["source_hash"] and cc.get("norm_id") == norm_id:
                        cached = cc
                except (OSError, ValueError, EOFError):
                    cached = None  # damaged cache entry: reprocess
            if cached:
                hits += 1
                payload = cached
            else:
                cues, invalid = N.parse_vtt(raw.decode("utf-8", errors="replace"))
                tokens, runs, duration = N.normalize_cues(cues, norm_cfg)
                if not tokens:
                    raise ValueError("no speech text after normalization")
                payload = {"source_hash": rec["source_hash"], "norm_id": norm_id,
                           "normalized_hash": C.sha256_text(" ".join(tokens)),
                           "duration": duration, "invalid_cues": invalid,
                           "tokens": tokens, "runs": runs}
                _save_doc_cache(ctx, c["video_id"], payload)
                processed += 1
                if invalid:
                    diagnostics.append({"stage": "ingest", "source_path": rel, "status": "warning",
                                        "message": "%d cues with invalid timing were ignored" % invalid})
            rec["normalized_hash"] = payload["normalized_hash"]
            rec["word_count"] = len(payload["tokens"])
            rec["duration_seconds"] = round(payload["duration"], 3)
        except Exception as exc:  # one bad file must not stop the rest
            rec["processing_status"] = "error: %s" % exc
            diagnostics.append({"stage": "ingest", "source_path": rel, "status": "error",
                                "message": str(exc)})
            ctx.log.warning("ingest [%s]: %s", rel, exc)
        documents.append(rec)
        if i % 25 == 0 or i == len(chosen):
            ctx.log.info("ingest: %d/%d documents", i, len(chosen))

    chash = corpus_hash(documents)
    manifest = {
        "stage": "ingest", "pipeline_version": C.PIPELINE_VERSION, "executed_at": C.now_iso(),
        "corpus_hash": chash,
        "normalize_config": norm_cfg, "normalize_id": norm_id,
        "corpus_scope": {
            "basis": "downloaded VTT files in the working directory only",
            "channel_history_complete": False,
            "note": "Members-only, deleted, unavailable or otherwise inaccessible videos are not "
                    "represented as analyzed documents; the full channel history is not known to this pipeline.",
        },
        "documents": documents,
        "included_video_ids": sorted(d["video_id"] for d in documents if d["processing_status"] == "ok"),
        "diagnostics": diagnostics,
    }
    C.write_json(ctx.ingest_manifest_path(), manifest)
    ctx.log.info("ingest: %d ok, %d errors, %d cache hits, corpus %s",
                 sum(d["processing_status"] == "ok" for d in documents),
                 sum(d["processing_status"] != "ok" for d in documents), hits, chash[:8])
    return {"documents": documents, "diagnostics": diagnostics, "cache_hits": hits,
            "processed": processed, "corpus_hash": chash}
