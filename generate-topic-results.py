#!/usr/bin/env python3
"""Generate an empirical Markdown topic report from NLP exports."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_cell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", default="categorization-results"
    )
    parser.add_argument(
        "--output", default="TOPIC_RESULTS.md"
    )
    args = parser.parse_args()

    results = Path(args.results)
    latest = results / "latest_run.txt"

    if latest.exists():
        run_name = latest.read_text(encoding="utf-8").strip()
        run_dir = results / run_name
    else:
        runs = sorted(p for p in results.iterdir() if p.is_dir())
        if len(runs) != 1:
            raise SystemExit(
                "Cannot select run automatically; specify latest_run.txt."
            )
        run_dir = runs[0]

    if not run_dir.is_dir():
        raise SystemExit(f"Missing run directory: {run_dir}")

    topics = read_csv(run_dir / "topics.csv")
    stability = read_csv(run_dir / "stability.csv")

    if not topics:
        raise SystemExit("No topics found.")

    grouped = defaultdict(list)
    model_ids = {}

    for topic in topics:
        k = int(topic["n_topics"])
        grouped[k].append(topic)
        model_ids[k] = topic["model_id"]

    lines = [
        "# Results by Topic",
        "",
        f"**Analytical run:** `{run_dir.name}`",
        "",
        "The following tables reproduce the topic identifiers, "
        "highest-weighted terms, dominant passage counts, and "
        "prominent lecture counts recorded by the NLP pipeline.",
        "",
        "Topic labels are not inferred from the vocabulary. "
        "An empty label in the source export remains unlabeled.",
        "",
        "The lecture counts use the pipeline's existing "
        "`n_lectures_prominent` definition. They do not "
        "represent ideological agreement or endorsement.",
        "",
    ]

    for k in sorted(grouped):
        model_id = model_ids[k]
        rows = sorted(
            grouped[k], key=lambda x: x["topic_id"]
        )

        lines.extend([
            f"## {k}-component model",
            "",
            f"**Model:** `{model_id}`",
            "",
            "| Topic | Representative terms | Dominant passages "
            "| Prominent lectures |",
            "|---|---|---:|---:|",
        ])

        for row in rows:
            terms = ", ".join(
                row["top_terms"].split(";")[:8]
            )

            lines.append(
                f"| {markdown_cell(row['topic_id'])} "
                f"| {markdown_cell(terms)} "
                f"| {row['n_passages_dominant']} "
                f"| {row['n_lectures_prominent']} |"
            )

        lines.extend(["", "### Model stability", ""])

        matched = [
            row for row in stability
            if row.get("model_a") == model_id
        ]

        if matched:
            lines.extend([
                "| Comparison | Metric | Value |",
                "|---|---|---:|",
            ])

            for row in matched:
                lines.append(
                    f"| {markdown_cell(row['comparison'])} "
                    f"| {markdown_cell(row['metric'])} "
                    f"| {markdown_cell(row['value'])} |"
                )
        else:
            lines.append(
                "No stability measurements were recorded "
                "for this model."
            )

        lines.extend([
            "",
            "These stability statistics describe comparisons "
            "between fitted models or partitions. They should "
            "not be interpreted as individual topic confidence "
            "scores unless the metric explicitly supports "
            "that interpretation.",
            "",
        ])

    lines.extend([
        "## Interpretive limitations",
        "",
        "The topics are unsupervised statistical components, "
        "not verified intellectual categories. Some components "
        "may capture promotional material, conversational "
        "expressions, or transcription artifacts.",
        "",
        "Prominent lecture counts can overlap because a lecture "
        "may be associated with several components. These "
        "counts should not be summed to estimate the number "
        "of distinct lectures.",
        "",
        "The representative terms describe the fitted model "
        "but do not establish endorsement, historical "
        "influence, or causal relationships.",
        "",
    ])

    output = Path(args.output)
    output.write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"Run: {run_dir.name}")
    print(f"Models: {', '.join(map(str, sorted(grouped)))}")
    print(f"Topics: {sum(map(len, grouped.values()))}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
