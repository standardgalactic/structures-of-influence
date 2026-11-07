"""VTT parsing and transcript normalization.

Source subtitle files are never modified. Normalization removes timestamps, markup
and rolling-caption overlap while retaining genuine repetitions, digressions and
promotional speech (the faithful transcript).
"""
import html
import re

from nlp.common import SOUND_TAG_RE

_TS = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})$")
_TAG = re.compile(r"<[^>]*>")


def parse_ts(s):
    m = _TS.match(s.strip())
    if not m:
        return None
    h, mi, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(sec) + int(ms) / 1000.0


def parse_vtt(text):
    """Return (cues, n_invalid). Each cue is (start, end, payload_lines).

    Raises ValueError for files that are not usable WebVTT.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or not lines[0].startswith("WEBVTT"):
        raise ValueError("missing WEBVTT header")
    blocks, cur = [], []
    for ln in lines[1:]:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    cues, invalid = [], 0
    for b in blocks:
        idx = next((i for i, l in enumerate(b) if "-->" in l), None)
        if idx is None:
            continue  # NOTE, STYLE, header metadata
        left, _, right = b[idx].partition("-->")
        right = right.strip().split()
        s = parse_ts(left)
        e = parse_ts(right[0]) if right else None
        if s is None or e is None or e < s:
            invalid += 1
            continue
        cues.append((s, e, b[idx + 1:]))
    if not cues:
        raise ValueError("no valid cues")
    return cues, invalid


def clean_line(line, strip_sound_tags=True):
    line = _TAG.sub("", line)
    line = html.unescape(line)
    if strip_sound_tags:
        line = SOUND_TAG_RE.sub(" ", line)
    return [t for t in line.split() if t != ">>"]


def _overlap(tail, cue, max_k):
    for k in range(min(len(tail), len(cue), max_k), 0, -1):
        if tail[-k:] == cue[:k]:
            return k
    return 0


def normalize_cues(cues, cfg):
    """Return (tokens, runs, duration).

    runs is a list of [start, end, n_tokens] giving the cue that produced each
    consecutive group of tokens, so every token keeps a timestamp.
    """
    min_overlap = cfg["min_overlap_tokens"]
    window = cfg["max_overlap_window"]
    dup_secs = cfg["duplicate_cue_max_seconds"]
    tokens, runs = [], []
    prev_two = None  # lines of the most recent two-line cue (rolling-caption signature)
    duration = 0.0
    for s, e, payload in cues:
        duration = max(duration, e)
        lines = [clean_line(l, cfg["strip_sound_tags"]) for l in payload]
        lines = [l for l in lines if l]
        if not lines:
            continue
        new_lines = lines
        if len(lines) == 2 and prev_two is not None and lines[0] == prev_two[1]:
            new_lines = [lines[1]]  # first line is the previous cue's second line
        if len(lines) == 2:
            prev_two = lines
        cue_tokens = [t for l in new_lines for t in l]
        k = _overlap(tokens[-window:], cue_tokens, window)
        if k and (k >= min_overlap or (k == len(cue_tokens) and (e - s) <= dup_secs)):
            cue_tokens = cue_tokens[k:]
        if not cue_tokens:
            continue
        tokens.extend(cue_tokens)
        if runs and runs[-1][0] == s and runs[-1][1] == e:
            runs[-1][2] += len(cue_tokens)
        else:
            runs.append([s, e, len(cue_tokens)])
    return tokens, runs, duration


def token_times(runs):
    """Expand runs into per-token (start, end) lists."""
    starts, ends = [], []
    for s, e, n in runs:
        starts.extend([s] * n)
        ends.extend([e] * n)
    return starts, ends
