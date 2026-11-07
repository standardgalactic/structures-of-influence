#!/usr/bin/env python3
"""Evidence-preserving YouTube VTT summarizer using Ollama."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

API = "http://localhost:11434/api/generate"

CHUNK_PROMPT = """You are analyzing a transcript of an academic or political lecture.

Reconstruct the speaker's actual arguments, not what you expect
the named historical thinker to have argued.

Identify:
1. Claims and the reasoning offered in their support.
2. Named thinkers, works, concepts, and historical examples.
3. Assumptions required for the argument.
4. Qualifications, objections, and unresolved questions.
5. Distinctions between the speaker's interpretation and
   claims explicitly attributed to historical sources.

Preserve concrete evidence and approximate transcript timestamps.
Do not introduce external facts or invent missing arguments.
Do not treat rhetorical assertions as established evidence.

Write detailed academic prose with concise section headings.
"""

SYNTHESIS_PROMPT = """You are synthesizing sequential analytical notes
from one lecture.

Produce a detailed, faithful argument reconstruction with sections:

Central Thesis
Argument Reconstruction
Intellectual Sources and Concepts
Evidence and Illustrations
Implicit Premises and Inferential Gaps
Qualifications and Counterarguments
Relation to the Study of Institutional and Cultural Influence

Separate what the lecturer explicitly claims from your own
interpretive inferences. Do not independently verify historical
claims or introduce outside information.

Use coherent academic prose, not bullet points.
Retain transcript timestamps where available.
"""

def ollama(prompt, model, retries=3):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_ctx": 8192
        }
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                API,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=1200) as response:
                result = json.load(response)
            return result["response"].strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def parse_vtt(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text)
    cues = []
    previous = ""

    for block in blocks:
        lines = block.strip().splitlines()
        timing = next(
            (line for line in lines if "-->" in line),
            None
        )
        if not timing:
            continue

        idx = lines.index(timing)
        content = " ".join(lines[idx + 1:])
        content = re.sub(r"<[^>]+>", "", content)
        content = re.sub(r"\{\\an\d+\}", "", content)
        content = re.sub(r"\s+", " ", content).strip()

        if not content:
            continue

        # Handle rolling auto-caption windows and exact duplicates.
        if content == previous:
            continue
        if previous and content.startswith(previous):
            content = content[len(previous):].strip()
        elif previous and previous.endswith(content):
            continue

        if not content:
            continue

        timestamp = timing.split("-->")[0].strip()
        cues.append((timestamp, content))
        previous = " ".join(lines[idx + 1:])
        previous = re.sub(r"<[^>]+>", "", previous)
        previous = re.sub(r"\s+", " ", previous).strip()

    return cues


def chunks_from_cues(cues, max_chars=12000):
    chunks = []
    current = []
    size = 0

    for timestamp, content in cues:
        line = f"[{timestamp}] {content}"
        if current and size + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def video_files(directory):
    selected = {}

    for path in sorted(directory.glob("*.vtt")):
        match = re.match(
            r"^(\d{8})_([A-Za-z0-9_-]{11})_(.+)\.(en-orig|en)\.vtt$",
            path.name
        )
        if not match:
            continue

        date, video_id, title, language = match.groups()
        priority = 0 if language == "en-orig" else 1

        if video_id not in selected or priority < selected[video_id][0]:
            selected[video_id] = (priority, path, date, title)

    return [
        (video_id, path, date, title)
        for video_id, (_, path, date, title) in sorted(
            selected.items(), key=lambda item: item[1][2]
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="transcripts")
    parser.add_argument("--output", default="analyses")
    parser.add_argument("--model", default="granite4.1:3b")
    parser.add_argument("--chunk-size", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    videos = video_files(source)
    if args.limit:
        videos = videos[:args.limit]

    print(f"Found {len(videos)} distinct videos.", flush=True)

    for number, (video_id, path, date, title) in enumerate(videos, 1):
        destination = output / f"{date}_{video_id}.md"
        if destination.exists():
            print(f"[{number}] SKIP {title}", flush=True)
            continue

        print(f"[{number}] {title}", flush=True)
        cues = parse_vtt(path)
        chunks = chunks_from_cues(cues, args.chunk_size)

        if not chunks:
            print("  No readable subtitle cues.", flush=True)
            continue

        notes = []
        for index, chunk in enumerate(chunks, 1):
            print(f"  Chunk {index}/{len(chunks)}", flush=True)
            prompt = (
                CHUNK_PROMPT
                + f"\nLecture: {title}\n"
                + f"Section {index} of {len(chunks)}\n\n"
                + chunk
            )
            notes.append(
                f"SECTION {index}\n"
                + ollama(prompt, args.model)
            )

        # Reduce notes hierarchically if necessary.
        while len("\n\n".join(notes)) > 24000:
            grouped = []
            for i in range(0, len(notes), 2):
                combined = "\n\n".join(notes[i:i + 2])
                grouped.append(
                    ollama(
                        "Consolidate these lecture notes faithfully. "
                        "Preserve claims, evidence, qualifications, "
                        "and timestamps. Do not invent facts.\n\n"
                        + combined,
                        args.model
                    )
                )
            notes = grouped

        synthesis = ollama(
            SYNTHESIS_PROMPT
            + f"\nLecture: {title}\n\n"
            + "\n\n".join(notes),
            args.model
        )

        document = (
            f"# {title}\n\n"
            f"**Publication date:** {date}\n\n"
            f"**Source:** https://www.youtube.com/watch?v={video_id}\n\n"
            f"**Transcript:** `{path.name}`\n\n"
            f"**Model:** `{args.model}`\n\n"
            f"{synthesis}\n"
        )

        destination.write_text(document, encoding="utf-8")
        print(f"  Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
