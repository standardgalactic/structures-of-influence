"""Evidence retrieval: representative passages per topic and phrase search."""
import re

from nlp import common as C

CONC_FIELDS = ["run_id", "model_id", "topic_id", "rank", "passage_id", "video_id", "title",
               "upload_date", "start_seconds", "weight", "youtube_url", "text"]


def search_phrase(passages, docs_by_id, phrase, video_id=None, case_sensitive=False, whole_word=True):
    """Exact phrase search over faithful passage text.

    Returns (hits, corpus_frequency, distinct_documents). Corpus frequency counts
    every occurrence; distinct_documents counts lectures with at least one.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    body = r"\s+".join(re.escape(w) for w in phrase.split())
    pattern = re.compile((r"\b%s\b" if whole_word else "%s") % body, flags)
    hits, total, docs = [], 0, set()
    for p in passages:
        if video_id and p["video_id"] != video_id:
            continue
        ms = list(pattern.finditer(p["text"]))
        if not ms:
            continue
        total += len(ms)
        docs.add(p["video_id"])
        d = docs_by_id.get(p["video_id"], {})
        for m in ms:
            a, b = max(0, m.start() - 120), min(len(p["text"]), m.end() + 120)
            hits.append({"passage_id": p["passage_id"], "video_id": p["video_id"],
                         "title": d.get("title", ""), "upload_date": d.get("upload_date", ""),
                         "start_seconds": p["start_seconds"],
                         "context": p["text"][a:b], "passage_text": p["text"],
                         "youtube_url": C.youtube_url(p["video_id"], p["start_seconds"])})
    return hits, total, len(docs)
