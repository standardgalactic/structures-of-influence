#!/usr/bin/env python3
"""Unified entry point for the Structures of Influence categorization pipeline.

Examples:
  python3 nlp/pipeline.py ingest
  python3 nlp/pipeline.py segment
  python3 nlp/pipeline.py fit-nmf --topics 12 20 30
  python3 nlp/pipeline.py similarities --method tfidf
  python3 nlp/pipeline.py stability
  python3 nlp/pipeline.py concordances
  python3 nlp/pipeline.py export
  python3 nlp/pipeline.py search "iron law" --video VIDEO_ID
  python3 nlp/pipeline.py run-all --limit 20      # limited test run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import common as C  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="project root containing transcripts/, corpus-cache/, and analysis outputs")
    ap.add_argument("--config", help="JSON file overriding default configuration")
    ap.add_argument("--dry-run", action="store_true", help="report planned work without writing")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--limit", type=int, help="process only the first N videos (date, video id order)")
    ap.add_argument("--video-ids", nargs="+", help="process only these video ids")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="parse and normalize VTT files, record corpus manifest")
    sub.add_parser("segment", help="split transcripts into passages and flag boilerplate")
    p = sub.add_parser("fit-nmf", help="fit passage-level NMF models")
    p.add_argument("--topics", nargs="+", type=int, help="topic counts (default 12 20 30)")
    p.add_argument("--force", action="store_true", help="refit even if cached")
    p = sub.add_parser("classify-frozen", help="classify current passages with an existing fitted model")
    p.add_argument("--model-id", required=True)
    p = sub.add_parser("similarities", help="document similarity")
    p.add_argument("--method", default="tfidf", choices=["tfidf"])
    sub.add_parser("stability", help="repeat fits and cross-model comparisons")
    sub.add_parser("concordances", help="print representative passages for each topic of the latest models")
    sub.add_parser("export", help="write categorization-results/RUN_ID")
    p = sub.add_parser("search", help="exact phrase search with frequency and distinct-document counts")
    p.add_argument("phrase")
    p.add_argument("--video")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--max", type=int, default=10)
    p = sub.add_parser("run-all", help="ingest, segment, fit-nmf, similarities, stability, export")
    p.add_argument("--topics", nargs="+", type=int)
    p.add_argument("--skip-stability", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = C.Ctx(args.root, C.load_config(args.config), dry_run=args.dry_run, verbose=args.verbose,
                limit=args.limit, video_ids=args.video_ids)
    from nlp import ingest, segment, topics_nmf, similarities, stability, export, concordance
    try:
        cmd = args.command
        if cmd == "ingest":
            ingest.run_ingest(ctx)
        elif cmd == "segment":
            segment.run_segment(ctx)
        elif cmd == "fit-nmf":
            topics_nmf.fit_all(ctx, args.topics, args.force)
        elif cmd == "classify-frozen":
            topics_nmf.classify_frozen(ctx, args.model_id)
        elif cmd == "similarities":
            similarities.run_similarities(ctx, args.method)
        elif cmd == "stability":
            _stability(ctx, stability, export)
        elif cmd == "concordances":
            _concordances(ctx)
        elif cmd == "export":
            export.run_export(ctx)
        elif cmd == "search":
            _search(ctx, args, segment, ingest, concordance)
        elif cmd == "run-all":
            ingest.run_ingest(ctx)
            segment.run_segment(ctx)
            topics_nmf.fit_all(ctx, args.topics)
            similarities.run_similarities(ctx)
            if not args.skip_stability:
                _stability(ctx, stability, export)
            export.run_export(ctx)
    except C.StageError as exc:
        ctx.log.error("%s", exc)
        return 2
    return 0


def _stability(ctx, stability, export):
    rows = stability.run_stability(ctx)
    if ctx.dry_run:
        return
    ing = C.read_json(ctx.ingest_manifest_path())
    sm = C.read_json(ctx.segment_manifest_path())
    C.write_json(export.stability_cache_path(ctx, ing, sm), rows)
    ctx.log.info("stability: %d comparison rows cached; run export to include them", len(rows))


def _concordances(ctx):
    from nlp import segment, topics_nmf
    import numpy as np
    passages, sm = segment.load_passages(ctx)
    ing = C.read_json(ctx.ingest_manifest_path())
    elig = topics_nmf.eligible(passages)
    ncfg = ctx.cfg["nmf"]
    for mid, r in sorted(ctx.load_models_registry().items(), key=lambda kv: kv[1].get("K", 0)):
        if r["kind"] != "refit" or r["corpus_hash"] != ing["corpus_hash"]:
            continue
        W = np.load(ctx.root / r["weights_file"])
        print("== %s (K=%d)" % (mid, r["K"]))
        for k in range(r["K"]):
            idx = topics_nmf.representative_indices(W, [p["video_id"] for p in elig], k,
                                                    ncfg["representative_passages"], ncfg["max_per_lecture"])
            print("T%02d: %s" % (k, ", ".join("%s@%ds" % (elig[i]["video_id"], elig[i]["start_seconds"])
                                              for i in idx)))


def _search(ctx, args, segment, ingest, concordance):
    passages, _ = segment.load_passages(ctx)
    docs = {d["video_id"]: d for d in C.read_json(ctx.ingest_manifest_path())["documents"]}
    hits, total, ndocs = concordance.search_phrase(passages, docs, args.phrase, args.video, args.case_sensitive)
    print("occurrences: %d   distinct lectures: %d" % (total, ndocs))
    for h in hits[: args.max]:
        print("- %s | %s | %s\n  ...%s...\n  %s" % (h["upload_date"], h["title"], h["video_id"],
                                                 h["context"], h["youtube_url"]))


if __name__ == "__main__":
    sys.exit(main())
