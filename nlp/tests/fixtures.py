"""Synthetic fixtures: YouTube-style rolling-caption VTT files and a themed toy corpus.

Everything here is invented text used only to test the software. None of it is
corpus data and none of it should be read as a finding.
"""
import random

SYL = ["ka", "to", "ri", "mu", "sen", "dal", "vor", "pi", "lun", "bex", "zor", "qui", "fan", "gor", "hal", "nim"]
COMMON = ("the of and to a in that it is was for on are with as be at this have from or by one had not but what "
          "all were when we there can an your which their said if do will each about how up out them then she "
          "many some so these would other into has more her two like him see time could no make than first been "
          "its who now people my made over did down only way find use may water long little very after words "
          "called just where most know").split()


def make_words(rng, n):
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(SYL) for _ in range(rng.choice([2, 3]))))
    return sorted(out)


def themed_tokens(rng, theme_words, mix, n_words):
    """mix: list of (theme_index, probability). Sentences end with a period."""
    toks = []
    while len(toks) < n_words:
        sent = []
        for _ in range(rng.randint(8, 16)):
            if rng.random() < 0.45:
                sent.append(rng.choice(COMMON))
            else:
                r, acc = rng.random(), 0.0
                for ti, pr in mix:
                    acc += pr
                    if r <= acc:
                        sent.append(rng.choice(theme_words[ti]))
                        break
                else:
                    sent.append(rng.choice(theme_words[mix[-1][0]]))
        sent[-1] += "."
        toks.extend(sent)
    return toks[:n_words]


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%06.3f" % (h, m, s)


def rolling_vtt(tokens, words_per_line=6, secs_per_line=2.0, dup_cues=True):
    """Render tokens as YouTube-style rolling auto-captions (two-line cues, tagged second line,
    plus the near-zero-length cue that repeats the last line)."""
    lines = [tokens[i:i + words_per_line] for i in range(0, len(tokens), words_per_line)]
    out = ["WEBVTT", "Kind: captions", "Language: en", ""]
    t = 0.0
    for i, line in enumerate(lines):
        start, end = t, t + secs_per_line
        tagged = "".join("<c> %s</c>" % w if j else "%s<c></c>" % w for j, w in enumerate(line))
        text = (" ".join(lines[i - 1]) + "\n" if i else "") + tagged
        out += ["%s --> %s align:start position:0%%" % (ts(start), ts(end)), text, ""]
        if dup_cues:
            out += ["%s --> %s align:start position:0%%" % (ts(end), ts(end + 0.01)), " ".join(line), ""]
        t = end
    return "\n".join(out) + "\n"


def write_vtt(directory, date, vid, title, tokens, variant="en-orig", **kw):
    name = "%s_%s_%s.%s.vtt" % (date, vid, title, variant)
    transcript_dir = directory / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / name).write_text(rolling_vtt(tokens, **kw), encoding="utf-8")
    return name


def vid_id(i):
    return ("vid%08d" % i)[:11]


def make_corpus(directory, n_videos=12, n_words=1800, n_themes=3, seed=1, two_variants=True):
    """Videos 0..n-1; video i is dominated by theme i % n_themes with a 15% secondary theme."""
    rng = random.Random(seed)
    theme_words = [make_words(rng, 40) for _ in range(n_themes)]
    meta = []
    for i in range(n_videos):
        main = i % n_themes
        sec = (i + 1) % n_themes
        toks = themed_tokens(rng, theme_words, [(main, 0.85), (sec, 0.15)], n_words)
        date = "2025%02d%02d" % (1 + i // 28, 1 + i % 28)
        name = write_vtt(directory, date, vid_id(i), "Lecture %d" % i, toks)
        if two_variants:
            write_vtt(directory, date, vid_id(i), "Lecture %d" % i, toks, variant="en")
        meta.append({"video_id": vid_id(i), "theme": main, "tokens": toks, "file": name})
    return meta, theme_words
