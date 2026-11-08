#!/usr/bin/env python3
"""Build a standalone, searchable HTML catalogue of lecture analyses."""

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path


def extract_record(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")

    heading = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    title = heading.group(1).strip() if heading else path.stem

    match = re.match(r"^(\d{8})_([A-Za-z0-9_-]{11})$", path.stem)
    if not match:
        return None

    date_raw, video_id = match.groups()

    try:
        date = datetime.strptime(date_raw, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        date = date_raw

    source = re.search(
        r"^\*\*Source:\*\*\s*(https?://\S+)",
        text,
        re.MULTILINE,
    )

    sections = {}
    current = None

    for line in text.splitlines():
        section = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
        if section:
            current = section.group(1).strip()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)

    sections = {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
    }

    thesis = next(
        (
            value
            for name, value in sections.items()
            if "central thesis" in name.lower()
        ),
        "",
    )

    if not thesis:
        body = re.sub(r"(?m)^#{1,6}\s+.*$", "", text)
        body = re.sub(r"(?m)^\*\*[^*]+:\*\*.*$", "", body)
        thesis = body.strip()

    return {
        "date": date,
        "video_id": video_id,
        "title": title,
        "source": source.group(1) if source else (
            f"https://www.youtube.com/watch?v={video_id}"
        ),
        "filename": path.name,
        "thesis": thesis,
        "sections": sections,
        "text": text,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structures of Influence | Lecture Catalogue</title>
<style>
:root {
  color-scheme: dark;
  --bg:#101512;
  --panel:#18201b;
  --line:#304236;
  --fg:#d5e3d8;
  --muted:#8ea493;
  --accent:#a4d5ae;
}
* { box-sizing:border-box; }
body {
  margin:0;
  background:var(--bg);
  color:var(--fg);
  font:15px/1.65 system-ui,-apple-system,Segoe UI,sans-serif;
}
header {
  border-bottom:1px solid var(--line);
  padding:32px max(24px,calc((100vw - 1200px)/2));
}
h1 {
  margin:0;
  font-size:clamp(24px,3vw,38px);
  font-weight:500;
  letter-spacing:-.035em;
}
.subtitle { color:var(--muted); margin:6px 0 0; }
.site-nav {
  display:flex;
  align-items:center;
  gap:12px;
  margin-top:16px;
  font-size:14px;
  color:var(--muted);
}
.site-nav a {
  color:var(--accent);
  text-decoration:none;
}
.site-nav a:hover {
  text-decoration:underline;
}
main {
  max-width:1200px;
  margin:auto;
  padding:24px;
}
.controls {
  display:flex;
  gap:12px;
  flex-wrap:wrap;
  margin-bottom:14px;
}
input,select {
  background:var(--panel);
  color:var(--fg);
  border:1px solid var(--line);
  border-radius:6px;
  padding:12px;
  font:inherit;
}
input { flex:1; min-width:220px; }
select { min-width:170px; }
#count { color:var(--muted); margin-bottom:18px; }
.layout {
  display:grid;
  grid-template-columns:minmax(260px,390px) minmax(0,1fr);
  gap:20px;
  align-items:start;
}
#results {
  max-height:75vh;
  overflow:auto;
  border:1px solid var(--line);
  border-radius:8px;
}
.item {
  width:100%;
  display:block;
  text-align:left;
  background:transparent;
  border:0;
  border-bottom:1px solid var(--line);
  color:var(--fg);
  padding:14px 16px;
  cursor:pointer;
  font:inherit;
}
.item:hover,.item.active { background:var(--panel); }
.item strong { display:block; font-weight:550; }
.meta { color:var(--muted); font-size:12px; }
#detail {
  border:1px solid var(--line);
  border-radius:8px;
  padding:24px;
  min-height:300px;
  overflow-wrap:anywhere;
}
#detail h2 { margin:0 0 10px; line-height:1.25; }
#detail h3 {
  margin:28px 0 10px;
  font-size:16px;
  color:var(--accent);
}
#detail .markdown-body {
  line-height:1.8;
  font-size:15px;
}
#detail .markdown-body p {
  white-space:normal;
  margin:0 0 1.2em;
}
#detail .markdown-body h2,
#detail .markdown-body h3,
#detail .markdown-body h4 {
  color:var(--accent);
  margin:1.8em 0 .65em;
  line-height:1.35;
}
#detail .markdown-body ul,
#detail .markdown-body ol {
  padding-left:1.8em;
  margin:0 0 1.3em;
}
#detail .markdown-body li {
  margin:.4em 0;
}
#detail .markdown-body blockquote {
  border-left:2px solid var(--accent);
  padding-left:1.2em;
  margin:1.5em 0;
  color:var(--muted);
}
#detail .markdown-body code {
  font-family:ui-monospace,monospace;
  font-size:.9em;
}
#detail .markdown-body a {
  color:var(--accent);
}

a { color:var(--accent); }
.actions { display:flex; gap:18px; flex-wrap:wrap; margin:16px 0; }
.empty { color:var(--muted); }
@media(max-width:760px) {
  .layout { grid-template-columns:1fr; }
  #results { max-height:300px; }
}
</style>
</head>
<body>
<header>
  <h1>Structures of Influence</h1>
  <p class="subtitle">Lecture analyses · Chronological research catalogue</p>
  <nav class="site-nav">
    <a href="../">Corpus Explorer</a>
    <span aria-hidden="true">/</span>
    <span aria-current="page">Lecture Analyses</span>
  </nav>
</header>
<main>
  <div class="controls">
    <input id="search" type="search"
      placeholder="Search titles, arguments, thinkers, concepts...">
    <select id="sort">
      <option value="newest">Newest first</option>
      <option value="oldest">Oldest first</option>
      <option value="title">Title A–Z</option>
    </select>
  </div>
  <div id="count"></div>
  <div class="layout">
    <div id="results"></div>
    <article id="detail">
      <p class="empty">Select a lecture to read its analysis.</p>
    </article>
  </div>
</main>
<script src="assets/marked.min.js"></script>
<script src="assets/purify.min.js"></script>
<script id="catalogue-data" type="application/json">__DATA__</script>
<script>
"use strict";
const data = JSON.parse(
  document.getElementById("catalogue-data").textContent
);
const search = document.getElementById("search");
const sort = document.getElementById("sort");
const results = document.getElementById("results");
const detail = document.getElementById("detail");
const count = document.getElementById("count");

let selected = null;
let filtered = [];

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function show(record) {
  selected = record.video_id;
  detail.replaceChildren();

  detail.append(el("h2", record.title));
  detail.append(el("div", record.date + " · " + record.video_id, "meta"));

  const actions = el("div", undefined, "actions");
  const source = el("a", "Original video");
  source.href = record.source;
  source.target = "_blank";
  source.rel = "noopener noreferrer";
  actions.append(source);

  const local = el("a", "Markdown analysis");
  local.href = record.filename;
  actions.append(local);
  detail.append(actions);

  const content = el("div", undefined, "markdown-body");

  if (typeof marked !== "undefined" &&
      typeof DOMPurify !== "undefined") {
    let markdown = record.text;

    // Remove the original title and metadata, already shown above.
    markdown = markdown.replace(/^#\\s+.*(?:\\r?\\n)/, "");
    markdown = markdown.replace(
      /^\\*\\*(?:Publication date|Source|Transcript|Model):\\*\\*.*$/gm,
      ""
    );

    content.innerHTML = DOMPurify.sanitize(
      marked.parse(markdown)
    );
  } else {
    // Preserve readable text if the CDN is unavailable.
    content.textContent = record.text;
    content.style.whiteSpace = "pre-wrap";
  }

  detail.append(content);

  renderList();
}

function renderList() {
  results.replaceChildren();

  const fragment = document.createDocumentFragment();
  for (const record of filtered) {
    const button = el("button", undefined, "item");
    button.type = "button";
    if (selected === record.video_id) button.classList.add("active");
    button.append(el("strong", record.title));
    button.append(el("span", record.date + " · " + record.video_id, "meta"));
    button.addEventListener("click", () => show(record));
    fragment.append(button);
  }
  results.append(fragment);
}

function refresh() {
  const query = search.value.trim().toLocaleLowerCase();
  const words = query.split(/\s+/).filter(Boolean);

  filtered = data.filter(record => {
    const haystack = (
      record.title + "\n" +
      record.date + "\n" +
      record.video_id + "\n" +
      record.text
    ).toLocaleLowerCase();
    return words.every(word => haystack.includes(word));
  });

  if (sort.value === "title") {
    filtered.sort((a,b) => a.title.localeCompare(b.title));
  } else {
    filtered.sort((a,b) =>
      sort.value === "oldest"
        ? a.date.localeCompare(b.date)
        : b.date.localeCompare(a.date)
    );
  }

  count.textContent = filtered.length + " of " + data.length + " lectures";
  renderList();
}

search.addEventListener("input", refresh);
sort.addEventListener("change", refresh);
refresh();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="analyses")
    args = parser.parse_args()

    directory = Path(args.input)
    records = []

    for path in sorted(directory.glob("*.md")):
        record = extract_record(path)
        if record:
            records.append(record)

    if not records:
        raise SystemExit(f"No matching analyses found in {directory}")

    records.sort(key=lambda r: (r["date"], r["video_id"]))

    json_path = directory / "index.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = json.dumps(records, ensure_ascii=False)
    payload = payload.replace("<", "\\u003c")
    payload = payload.replace(">", "\\u003e")
    payload = payload.replace("&", "\\u0026")

    page = HTML_TEMPLATE.replace("__DATA__", payload)
    html_path = directory / "index.html"
    html_path.write_text(page, encoding="utf-8")

    print(f"Indexed {len(records)} lectures.")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
