#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/academic-agent"

echo "Preparing Structures of Influence repository..."

# Preserve the existing working layout.
mkdir -p docs

# Exclude source transcripts, large caches, local models,
# and machine-specific files from the public repository.
cat > .gitignore <<'EOF'
# Downloaded third-party transcripts
*.vtt
*.srt
*.ass
*.ttml
*.srv1
*.srv2
*.srv3

# Local research source material
downloads/
transcripts/
raw/
source-corpus/

# Cached feature matrices and fitted models
corpus-cache/
*.joblib
*.npy
*.npz
*.pkl
*.pickle

# Local generated lecture analyses
analyses/

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
*.egg-info/

# Downloader state and authentication
cookies.txt
*cookies*.txt
downloaded.txt
*.log

# Local inventories
file-list.txt

# System files
.DS_Store
Thumbs.db
EOF

# A minimal dependency-free landing document.
cat > README.md <<'EOF'
# Structures of Influence

A reproducible corpus-linguistic and computational analysis
of political, economic, historical, and cultural discourse.

The project investigates recurring concepts, intellectual
relationships, explanatory structures, and changes in
interpretive vocabulary across a growing collection of
publicly available lectures.

## Research methods

The analytical system combines lexical frequency analysis,
n-gram statistics, TF-IDF, passage-level topic modeling
using non-negative matrix factorization, document similarity,
concordance retrieval, and model comparison.

Topic assignments are statistical descriptions, not
determinations of ideological agreement, authorial intention,
or historical influence.

## Interfaces

The repository contains browser-based exploratory interfaces
for lexical analysis and corpus categorization.

The interfaces operate locally on exported analytical data.

## Reproducing the analysis

See [NLP documentation](nlp/README.md).

Install the Python dependencies:

    python3 -m pip install -r nlp/requirements.txt

Inspect the pipeline commands:

    python3 nlp/pipeline.py --help

Source transcripts are not redistributed in this repository.
Researchers must obtain source material independently and
comply with applicable copyright and platform requirements.

## Methodological limitations

The corpus represents available downloaded material rather
than a complete history of the source channel.

Automated transcripts may contain recognition errors,
repeated captions, and other artifacts.

Model-derived topics and similarity scores require
interpretation against the original passages.
EOF

echo
echo "Potentially large directories:"
du -sh corpus-cache corpus-results \
  categorization-results analyses 2>/dev/null || true

echo
echo "Repository preparation complete."
echo "No files were deleted or moved."
EOF
