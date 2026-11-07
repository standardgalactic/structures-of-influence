import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nlp import common as C
from nlp import concordance, export, ingest, normalize, segment, similarities, stability, topics_nmf
from nlp.tests import fixtures as fx


def small_cfg(**over):
    cfg = C.load_config()
    cfg["nmf"]["topics"] = [3]
    cfg["nmf"]["tfidf"]["min_df"] = 2
    cfg["nmf"]["tfidf"]["ngram_range"] = [1, 1]
    cfg["nmf"]["representative_passages"] = 4
    cfg["boilerplate"]["min_docs"] = 6
    cfg["stability"]["seeds"] = 1
    return C.deep_update(cfg, over)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def ctx(self, **kw):
        return C.Ctx(self.tmp, small_cfg(), **kw)


class IngestTests(Base):
    def test_1_two_variants_one_document(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=3, n_words=300)
        res = ingest.run_ingest(self.ctx())
        self.assertEqual(len(res["documents"]), 3)
        self.assertEqual({d["language_variant"] for d in res["documents"]}, {"en-orig"})
        self.assertEqual(sum(d["status"] == "variant_skipped" for d in res["diagnostics"]), 3)

    def test_2_overlap_does_not_inflate_word_count(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=2, n_words=900)
        res = ingest.run_ingest(self.ctx())
        for d, m in zip(res["documents"], meta):
            self.assertEqual(d["word_count"], len(m["tokens"]))
            payload = ingest.load_doc_cache(self.ctx(), d["video_id"])
            self.assertEqual(payload["tokens"], m["tokens"])

    def test_3_genuine_repetition_preserved(self):
        toks = ("we said no no no more than that and then very very good indeed it was the "
                "the end of the story I think I think so").split()
        fx.write_vtt(self.tmp, "20250101", "abcdefghijk", "Reps", toks, words_per_line=4)
        res = ingest.run_ingest(self.ctx())
        payload = ingest.load_doc_cache(self.ctx(), "abcdefghijk")
        self.assertEqual(payload["tokens"], toks)

    def test_4_malformed_vtt_does_not_stop_others(self):
        fx.make_corpus(self.tmp, n_videos=2, n_words=200, two_variants=False)
        (self.tmp / "transcripts" / "20250301_BADVIDEO123_Broken.en-orig.vtt").write_text("not a vtt file at all", encoding="utf-8")
        (self.tmp / "transcripts" / "20250302_EMPTYVIDEO1_Empty.en-orig.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
        res = ingest.run_ingest(self.ctx())
        statuses = {d["video_id"]: d["processing_status"] for d in res["documents"]}
        self.assertEqual(sum(s == "ok" for s in statuses.values()), 2)
        self.assertTrue(statuses["BADVIDEO123"].startswith("error"))
        self.assertTrue(statuses["EMPTYVIDEO1"].startswith("error"))
        self.assertTrue(any(d["status"] == "error" and "Broken" in d["source_path"] for d in res["diagnostics"]))

    def test_5_6_cache_reuse_and_reprocess(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=3, n_words=300, two_variants=False)
        r1 = ingest.run_ingest(self.ctx())
        self.assertEqual((r1["processed"], r1["cache_hits"]), (3, 0))
        r2 = ingest.run_ingest(self.ctx())
        self.assertEqual((r2["processed"], r2["cache_hits"]), (0, 3))   # unchanged: reuse
        fx.write_vtt(self.tmp, "20250101", meta[0]["video_id"], "Lecture 0",
                     meta[0]["tokens"] + ["extra", "words", "appended"])
        r3 = ingest.run_ingest(self.ctx())
        self.assertEqual((r3["processed"], r3["cache_hits"]), (1, 2))   # changed: reprocess one
        self.assertNotEqual(r3["corpus_hash"], r2["corpus_hash"])

    def test_title_and_timing(self):
        fx.write_vtt(self.tmp, "20250102", "abcdefghijk", "Why %s" % "？", ["one", "two", "three"] * 20)
        res = ingest.run_ingest(self.ctx())
        d = res["documents"][0]
        self.assertEqual(d["upload_date"], "2025-01-02")
        self.assertTrue(d["title"].endswith("?"))
        self.assertGreater(float(d["duration_seconds"]), 0)


class SegmentTests(Base):
    def test_passage_sizes_and_traceability(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=2, n_words=2000, two_variants=False)
        ctx = self.ctx()
        ingest.run_ingest(ctx)
        res = segment.run_segment(ctx)
        ps = res["passages"]
        for vid in {p["video_id"] for p in ps}:
            mine = [p for p in ps if p["video_id"] == vid]
            self.assertEqual(sum(p["word_count"] for p in mine), 2000)
            self.assertTrue(all(480 <= p["word_count"] <= 800 for p in mine))
            toks = next(m["tokens"] for m in meta if m["video_id"] == vid)
            self.assertEqual(" ".join(p["text"] for p in mine), " ".join(toks))
            starts = [p["start_seconds"] for p in mine]
            self.assertEqual(starts, sorted(starts))
        ids = [p["passage_id"] for p in ps]
        self.assertEqual(len(ids), len(set(ids)))
        # stable ids on a rerun, distinct segmentation version when configuration changes
        res2 = segment.run_segment(ctx)
        self.assertEqual(ids, [p["passage_id"] for p in res2["passages"]])
        ctx3 = C.Ctx(self.tmp, small_cfg(segment={"target_words": 400, "min_words": 300, "max_words": 500}))
        res3 = segment.run_segment(ctx3)
        self.assertNotEqual(res3["segmentation_version"], res["segmentation_version"])
        self.assertTrue(set(ids).isdisjoint(p["passage_id"] for p in res3["passages"]))

    def test_boilerplate_flagged_and_recorded(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=8, n_words=1300, two_variants=False)
        promo = ("please like and subscribe to the channel and join the members area for exclusive "
                 "early access videos and also support the work through the link in the description below "
                 "every single week without fail so that we can keep making more of this content for you ").split()
        for i in range(8):
            toks = promo * 14 + meta[i]["tokens"]   # ~670 words of identical promo at the start
            fx.write_vtt(self.tmp, "2025%02d%02d" % (1, 1 + i), meta[i]["video_id"], "Lecture %d" % i, toks)
        ctx = self.ctx()
        ingest.run_ingest(ctx)
        res = segment.run_segment(ctx)
        flagged = [p for p in res["passages"] if p["excluded_from_content_analysis"]]
        self.assertGreaterEqual(len(flagged), 8)
        self.assertTrue(all("cross_document_near_duplicate" in p["exclusion_reason"] for p in flagged))
        self.assertTrue(all(p["passage_index"] == 0 for p in flagged))
        rows = C.read_csv(Path(res["path"]))
        self.assertEqual(sum(r["excluded_from_content_analysis"] == "True" for r in rows), len(flagged))


class TopicTests(Base):
    def test_8_aggregation_numerically_correct(self):
        W = np.array([[2.0, 2.0, 0.0], [0.0, 1.0, 3.0], [5.0, 0.0, 0.0]])
        meta = [("a", 100), ("a", 300), ("b", 50)]
        agg = topics_nmf.aggregate_documents(W, meta)
        expect_a = (100 * np.array([.5, .5, 0]) + 300 * np.array([0, .25, .75])) / 400
        np.testing.assert_allclose(agg["a"]["weights"], expect_a)
        np.testing.assert_allclose(agg["b"]["weights"], [1, 0, 0])
        self.assertAlmostEqual(agg["a"]["weights"].sum(), 1.0)

    def test_9_zero_weight_passages_explicit(self):
        W = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        meta = [("a", 100), ("a", 900), ("z", 200)]
        agg = topics_nmf.aggregate_documents(W, meta)
        np.testing.assert_allclose(agg["a"]["weights"], [.5, .5])   # zero row left the denominator
        self.assertEqual(agg["a"]["unassigned_passages"], 1)
        self.assertEqual(agg["a"]["assigned_words"], 100)
        self.assertEqual(agg["z"]["assigned_words"], 0)
        np.testing.assert_allclose(agg["z"]["weights"], [0, 0])     # flagged, not invented

    def test_nmf_recovers_planted_themes_and_representatives_trace(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=12, n_words=1800, two_variants=False)
        ctx = self.ctx()
        ingest.run_ingest(ctx)
        seg = segment.run_segment(ctx)
        mids = topics_nmf.fit_all(ctx)
        self.assertEqual(len(mids), 1)
        reg = ctx.load_models_registry()
        W = np.load(ctx.root / reg[mids[0]]["weights_file"])
        elig = topics_nmf.eligible(seg["passages"])
        self.assertEqual(W.shape, (len(elig), 3))
        agg = topics_nmf.aggregate_documents(W, [(p["video_id"], p["word_count"]) for p in elig])
        dom = {m["video_id"]: int(np.argmax(agg[m["video_id"]]["weights"])) for m in meta}
        # planted theme -> discovered topic mapping must be consistent across lectures
        mapping = {}
        for m in meta:
            mapping.setdefault(m["theme"], set()).add(dom[m["video_id"]])
        self.assertTrue(all(len(v) == 1 for v in mapping.values()), mapping)
        self.assertEqual(len({next(iter(v)) for v in mapping.values()}), 3)
        # 11: representative passages come from the actual source text
        by_id = {p["passage_id"]: p for p in elig}
        toks_by_vid = {m["video_id"]: " ".join(m["tokens"]) for m in meta}
        for k in range(3):
            idx = topics_nmf.representative_indices(W, [p["video_id"] for p in elig], k, 4, 2)
            self.assertTrue(idx)
            for i in idx:
                self.assertIn(elig[i]["text"], toks_by_vid[elig[i]["video_id"]])

    def test_7_new_lecture_does_not_alter_frozen_model(self):
        meta, _ = fx.make_corpus(self.tmp, n_videos=9, n_words=1500, two_variants=False)
        ctx = self.ctx()
        ingest.run_ingest(ctx)
        segment.run_segment(ctx)
        mid = topics_nmf.fit_all(ctx)[0]
        reg = ctx.load_models_registry()
        mfile = ctx.root / reg[mid]["model_file"]
        before = C.sha256_bytes(mfile.read_bytes())
        W_before = np.load(ctx.root / reg[mid]["weights_file"]).copy()
        # add a lecture, re-ingest, re-segment, classify with the frozen model
        rng = fx.random.Random(99)
        theme_words = [fx.make_words(fx.random.Random(1), 40)]
        fx.write_vtt(self.tmp, "20250601", "newlecture1", "New", fx.themed_tokens(rng, theme_words, [(0, 1.0)], 1500))
        ingest.run_ingest(ctx)
        segment.run_segment(ctx)
        fid = topics_nmf.classify_frozen(ctx, mid)
        self.assertEqual(before, C.sha256_bytes(mfile.read_bytes()))
        np.testing.assert_array_equal(W_before, np.load(ctx.root / ctx.load_models_registry()[mid]["weights_file"]))
        self.assertNotEqual(fid, mid)
        Wf = np.load(ctx.root / ctx.load_models_registry()[fid]["weights_file"])
        self.assertGreater(Wf.shape[0], W_before.shape[0])
        # old passages classify identically to their original weights (same vectorizer, same transform)
        old_ids = C.read_json(ctx.features_dir / (mid + ".passage_ids.json"))
        new_ids = C.read_json(ctx.features_dir / (fid + ".passage_ids.json"))
        pos = {p: i for i, p in enumerate(new_ids)}
        shared = [i for i, p in enumerate(old_ids) if p in pos]
        self.assertGreater(len(shared), 0)
        # transform() re-solves each row; allow a documented numerical tolerance
        np.testing.assert_allclose(W_before[shared], Wf[[pos[old_ids[i]] for i in shared]], atol=5e-3)

    def test_determinism_same_inputs_same_outputs(self):
        fx.make_corpus(self.tmp, n_videos=9, n_words=1500, two_variants=False)
        ctx = self.ctx()
        ingest.run_ingest(ctx)
        segment.run_segment(ctx)
        mid = topics_nmf.fit_all(ctx)[0]
        reg = ctx.load_models_registry()
        W1 = np.load(ctx.root / reg[mid]["weights_file"]).copy()
        topics_nmf.fit_all(ctx, force=True)
        np.testing.assert_allclose(W1, np.load(ctx.root / reg[mid]["weights_file"]), atol=1e-9)


class SimilarityAndExportTests(Base):
    def run_all(self, with_stability=True):
        fx.make_corpus(self.tmp, n_videos=9, n_words=1500, two_variants=False)
        ctx = self.ctx()
        ingest.run_ingest(ctx)
        segment.run_segment(ctx)
        topics_nmf.fit_all(ctx)
        similarities.run_similarities(ctx)
        if with_stability:
            rows = stability.run_stability(ctx)
            ing, sm = C.read_json(ctx.ingest_manifest_path()), C.read_json(ctx.segment_manifest_path())
            C.write_json(export.stability_cache_path(ctx, ing, sm), rows)
        return ctx, export.run_export(ctx)

    def test_10_similarity_range_symmetry_and_theme_structure(self):
        ctx, run_id = self.run_all(with_stability=False)
        mid = next(m for m, r in ctx.load_models_registry().items() if r["kind"] == "similarity")
        S = np.load(ctx.features_dir / (mid + ".S.npy"))
        self.assertTrue(np.allclose(S, S.T))
        self.assertTrue((S >= 0).all() and (S <= 1 + 1e-9).all())
        np.testing.assert_allclose(np.diag(S), 1.0)
        vids = C.read_json(ctx.features_dir / (mid + ".vids.json"))
        # lectures on the same planted theme are more similar than lectures on different themes
        theme = {fx.vid_id(i): i % 3 for i in range(9)}
        same = [S[i, j] for i in range(9) for j in range(i + 1, 9) if theme[vids[i]] == theme[vids[j]]]
        diff = [S[i, j] for i in range(9) for j in range(i + 1, 9) if theme[vids[i]] != theme[vids[j]]]
        self.assertGreater(np.mean(same), np.mean(diff))
        rows = C.read_csv(ctx.results_dir / run_id / "document_similarity.csv")
        self.assertEqual(len(rows), 9 * 8 // 2)
        self.assertTrue(all(r["method"] == "tfidf_cosine" and r["model_id"] == mid for r in rows))

    def test_export_schema_manifest_and_run_id(self):
        ctx, run_id = self.run_all()
        out = ctx.results_dir / run_id
        expected = {"manifest.json", "documents.csv", "passages.csv", "topics.csv", "topic_terms.csv",
                    "passage_topics.csv", "document_topics.csv", "document_similarity.csv",
                    "topic_concordances.csv", "stability.csv", "report.json", "document_terms.csv"}
        self.assertEqual(expected, {p.name for p in out.iterdir()})
        for name in expected:
            if name.endswith(".csv"):
                raw = (out / name).read_bytes()
                raw.decode("utf-8")
                self.assertTrue(raw.split(b"\n", 1)[0].startswith(b"run_id"))
        man = C.read_json(out / "manifest.json")
        self.assertEqual(man["run_id"], run_id)
        self.assertEqual(len(man["corpus"]["included_documents"]), 9)
        for key in ("software", "preprocessing", "nmf_config", "models", "corpus"):
            self.assertIn(key, man)
        self.assertFalse(man["corpus"]["scope"]["channel_history_complete"])
        dt = C.read_csv(out / "document_topics.csv")
        per_doc = {}
        for r in dt:
            per_doc[r["video_id"]] = per_doc.get(r["video_id"], 0.0) + float(r["weight"])
        for v in per_doc.values():
            self.assertAlmostEqual(v, 1.0, places=3)
        tp = C.read_csv(out / "topics.csv")
        self.assertTrue(all(r["label"] == "" and r["topic_id"].startswith("T") for r in tp))
        conc = C.read_csv(out / "topic_concordances.csv")
        self.assertTrue(all(r["youtube_url"].startswith("https://www.youtube.com/watch?v=") and
                            r["youtube_url"].endswith("s") for r in conc))
        stab = C.read_csv(out / "stability.csv")
        self.assertTrue(any(r["comparison"] == "repeat_fit" for r in stab))

    def test_12_missing_report_is_unavailable_not_fabricated(self):
        ctx, run_id = self.run_all(with_stability=False)
        out = ctx.results_dir / run_id
        self.assertEqual(C.read_csv(out / "stability.csv"), [])
        rep = C.read_json(out / "report.json")
        self.assertEqual(rep["components"]["stability"], "unavailable")
        self.assertEqual(rep["components"]["embedding_similarity"], "unavailable")
        self.assertEqual(export.load_optional(self.tmp / "nope.json")["status"], "unavailable")

    def test_investigator_labels_kept_separate(self):
        ctx, run_id = self.run_all(with_stability=False)
        topics = C.read_csv(ctx.results_dir / run_id / "topics.csv")
        C.write_csv(ctx.results_dir / "topic_labels.csv", ["model_id", "topic_id", "label", "note"],
                    [{"model_id": topics[0]["model_id"], "topic_id": "T00", "label": "example label", "note": ""}])
        export.run_export(ctx)
        topics = C.read_csv(ctx.results_dir / run_id / "topics.csv")
        t0 = next(r for r in topics if r["topic_id"] == "T00")
        self.assertEqual((t0["label"], t0["label_source"]), ("example label", "investigator"))
        self.assertEqual(next(r for r in topics if r["topic_id"] == "T01")["label"], "")


class StabilityAndSearchTests(Base):
    def test_matching_and_metrics(self):
        rng = np.random.RandomState(0)
        H = rng.rand(5, 40)
        perm = rng.permutation(5)
        sims, _ = stability.match_topics(H, H[perm])
        np.testing.assert_allclose(sims, 1.0)
        self.assertAlmostEqual(stability.coverage(H, H[perm]), 1.0)
        a = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(stability.co_assignment(a, np.array([2, 2, 0, 0, 1, 1])), 1.0)

    def test_search_counts_frequency_and_distinct_documents(self):
        passages = [{"passage_id": "a", "video_id": "v1", "text": "the Iron Law of oligarchy and the iron law again", "start_seconds": 12.7},
                    {"passage_id": "b", "video_id": "v2", "text": "no mention here", "start_seconds": 0},
                    {"passage_id": "c", "video_id": "v1", "text": "an iron   law appears", "start_seconds": 70}]
        docs = {"v1": {"title": "T1", "upload_date": "2025-01-01"}}
        hits, total, ndocs = concordance.search_phrase(passages, docs, "iron law")
        self.assertEqual((total, ndocs), (3, 1))
        self.assertEqual(hits[0]["youtube_url"], "https://www.youtube.com/watch?v=v1&t=12s")
        h2, t2, _ = concordance.search_phrase(passages, docs, "iron law", case_sensitive=True)
        self.assertEqual(t2, 2)   # "iron law" lowercase occurs once in each of two passages


class NormalizeUnitTests(unittest.TestCase):
    def test_markup_entities_and_sound_tags(self):
        text = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<c>Tom &amp; Jerry</c> [Music] >> hello\n"
        cues, _ = normalize.parse_vtt(text)
        toks, runs, dur = normalize.normalize_cues(cues, C.DEFAULT_CONFIG["normalize"])
        self.assertEqual(toks, ["Tom", "&", "Jerry", "hello"])
        self.assertEqual(dur, 3.0)

    def test_invalid_cue_timing_ignored(self):
        text = "WEBVTT\n\n00:00:05.000 --> 00:00:03.000\nbackwards\n\n00:00:06.000 --> 00:00:08.000\nfine words\n"
        cues, invalid = normalize.parse_vtt(text)
        self.assertEqual((len(cues), invalid), (1, 1))


if __name__ == "__main__":
    unittest.main()
