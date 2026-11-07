"""Shared helpers: configuration, hashing, atomic writes, CSV I/O, run context."""
import copy
import csv
import datetime
import hashlib
import io
import json
import logging
import os
import platform
import re
import tempfile
from pathlib import Path

PIPELINE_VERSION = "0.1.0"

DEFAULT_CONFIG = {
    "normalize": {
        "version": "norm-1",
        # Minimum token overlap between consecutive cues before the overlap is
        # treated as a rolling-caption artifact and removed.
        "min_overlap_tokens": 3,
        "max_overlap_window": 200,
        # Cues this short (seconds) that repeat the preceding text are treated
        # as the zero-length duplicate cues YouTube emits.
        "duplicate_cue_max_seconds": 0.1,
        "strip_sound_tags": True,
    },
    "segment": {
        "version": "seg-1",
        "target_words": 650,
        "min_words": 500,   # soft limit: short lectures yield one short passage
        "max_words": 800,
        "boundary_slack": 120,
    },
    "boilerplate": {
        "enabled": True,
        "shingle_words": 8,
        "min_docs": 4,        # a shingle counts as shared if it occurs in this many lectures
        "min_fraction": 0.5,  # share of a passage's shingles that are shared
        # Optional lexical rules: {"name": str, "pattern": regex, "min_matches": int}
        "lexical_rules": [],
    },
    "nmf": {
        "topics": [12, 20, 30],
        "seed": 0,
        "init": "nndsvd",
        "max_iter": 400,
        "top_terms": 20,
        "representative_passages": 8,
        "max_per_lecture": 2,
        "prominent_min": 0.10,
        "tfidf": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_df": 0.85,
            "sublinear_tf": True,
            "norm": "l2",
            # "english", null (keep all words, so negation is retained), or a list.
            "stop_words": "english",
        },
    },
    "similarity": {
        "min_df": 2,
        "max_df": 0.85,
        "stop_words": "english",
        "nn_limit": 0,        # 0 exports all pairs; N exports N nearest neighbours per lecture
        "top_terms_per_document": 30,
    },
    "stability": {"seeds": 3},
}


def deep_update(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    extra = None
    if path:
        with open(path, encoding="utf-8") as fh:
            extra = json.load(fh)
    return deep_update(DEFAULT_CONFIG, extra)


def stable_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def cfg_hash(obj, n=8):
    return sha256_text(stable_json(obj))[:n]


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path, fieldnames, rows):
    """UTF-8 CSV with header row, written atomically."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    atomic_write_text(path, buf.getvalue())


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def package_versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "joblib"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def youtube_url(video_id, start_seconds):
    return "https://www.youtube.com/watch?v=%s&t=%ds" % (video_id, int(start_seconds))


class StageError(Exception):
    """Error carrying the processing stage and, where relevant, the source file."""

    def __init__(self, stage, message, source=None):
        self.stage = stage
        self.source = source
        loc = " [%s]" % source if source else ""
        super().__init__("%s%s: %s" % (stage, loc, message))


class Ctx:
    """Paths, configuration and run flags shared by every stage."""

    def __init__(self, root, config=None, dry_run=False, verbose=False, limit=None, video_ids=None):
        self.root = Path(root).expanduser().resolve()
        self.cfg = config or load_config()
        self.dry_run = dry_run
        self.limit = limit
        self.video_ids = set(video_ids) if video_ids else None
        self.cache = self.root / "corpus-cache"
        self.docs_dir = self.cache / "documents"
        self.passages_dir = self.cache / "passages"
        self.features_dir = self.cache / "features"
        self.models_dir = self.cache / "models"
        self.manifests_dir = self.cache / "manifests"
        self.results_dir = self.root / "categorization-results"
        self.log = logging.getLogger("nlp")
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.DEBUG if verbose else logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )

    def ensure_dirs(self):
        for d in (self.docs_dir, self.passages_dir, self.features_dir, self.models_dir,
                  self.manifests_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Manifests written by individual stages
    def ingest_manifest_path(self):
        return self.manifests_dir / "ingest.json"

    def segment_manifest_path(self):
        return self.manifests_dir / "segment.json"

    def models_registry_path(self):
        return self.manifests_dir / "models.json"

    def load_models_registry(self):
        p = self.models_registry_path()
        return read_json(p) if p.exists() else {}

    def save_models_registry(self, reg):
        write_json(self.models_registry_path(), reg)


SOUND_TAG_RE = re.compile(r"\[[^\]]*\]")
