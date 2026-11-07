#!/usr/bin/env python3
"""Lexical and n-gram analysis of Academic Agent VTT transcripts."""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)

# Retain negation and several words that can be analytically important.
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {
    "no", "not", "nor", "never", "against",
    "between", "without", "through", "under",
}

def read_vtt(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    cues = []
    previous = ""

    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().splitlines()
        index = next(
            (i for i, line in enumerate(lines) if "-->" in line),
            None
        )
        if index is None:
            continue

        content = " ".join(lines[index + 1:])
        content = re.sub(r"<[^>]*>", "", content)
        content = re.sub(r"\s+", " ", content).strip()

        if not content:
            continue

        # Remove repeated rolling-caption prefixes.
        if content == previous:
            continue
        if previous and content.startswith(previous):
            content = content[len(previous):].strip()
        elif previous and previous.endswith(content):
            continue

        if content:
            cues.append(content)

        previous = " ".join(lines[index + 1:])
        previous = re.sub(r"<[^>]*>", "", previous)
        previous = re.sub(r"\s+", " ", previous).strip()

    return " ".join(cues)


def select_transcripts(directory):
    chosen = {}

    for path in sorted(directory.glob("*.vtt")):
        match = re.match(
            r"^(\d{8})_([A-Za-z0-9_-]{11})_(.+)\.(en-orig|en)\.vtt$",
            path.name
        )
        if not match:
            continue

        date, video_id, title, language = match.groups()
        priority = 0 if language == "en-orig" else 1

        if video_id not in chosen or priority < chosen[video_id][0]:
            chosen[video_id] = (
                priority, date, title, path
            )

    return [
        (video_id, date, title, read_vtt(path))
        for video_id, (_, date, title, path) in sorted(
            chosen.items(),
            key=lambda item: item[1][1]
        )
    ]


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="transcripts")
    parser.add_argument("--output", default="corpus-results")
    parser.add_argument("--top", type=int, default=200)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    records = select_transcripts(Path(args.input))
    records = [r for r in records if r[3].strip()]

    if not records:
        raise SystemExit("No readable transcripts found.")

    ids, dates, titles, documents = map(list, zip(*records))

    print(f"Analyzing {len(documents)} transcripts.")

    metadata = [
        (video_id, date, title, len(text.split()))
        for video_id, date, title, text in records
    ]
    write_csv(
        output / "documents.csv",
        ["video_id", "date", "title", "word_count"],
        metadata,
    )

    for n in (1, 2, 3, 4):
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words=list(STOP_WORDS),
            ngram_range=(n, n),
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b",
        )
        matrix = vectorizer.fit_transform(documents)
        totals = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        ranked = sorted(
            zip(terms, totals),
            key=lambda item: (-item[1], item[0])
        )[:args.top]

        write_csv(
            output / f"{n}grams.csv",
            ["ngram", "frequency"],
            ranked,
        )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2 if len(documents) > 2 else 1,
        max_df=0.95 if len(documents) > 2 else 1.0,
    )

    matrix = tfidf.fit_transform(documents)
    terms = tfidf.get_feature_names_out()
    rows = []

    for i, row in enumerate(matrix):
        scores = row.toarray().ravel()
        top_indices = scores.argsort()[::-1][:30]

        for rank, index in enumerate(top_indices, 1):
            if scores[index] > 0:
                rows.append((
                    ids[i], dates[i], titles[i],
                    rank, terms[index], round(float(scores[index]), 6)
                ))

    write_csv(
        output / "distinctive_terms.csv",
        ["video_id", "date", "title", "rank", "term", "tfidf"],
        rows,
    )

    print(f"Results written to {output}/")


if __name__ == "__main__":
    main()
