#!/usr/bin/env bash
set -Eeuo pipefail

# Download English subtitles without downloading video or audio.
#
# Usage:
#   ./download-subtitles.sh CHANNEL_URL
#   ./download-subtitles.sh PLAYLIST_URL
#   ./download-subtitles.sh VIDEO_URL
#
# Example:
#   ./download-subtitles.sh \
#     'https://www.youtube.com/@AcademicAgent/videos'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/transcripts"
ARCHIVE="$ROOT/.subtitle-download-archive.txt"

URL="${1:-}"

if [[ -z "$URL" ]]; then
    echo "Usage: $0 CHANNEL_OR_PLAYLIST_URL" >&2
    exit 1
fi

if ! command -v yt-dlp >/dev/null 2>&1; then
    echo "yt-dlp is required." >&2
    echo "Install: python3 -m pip install -U yt-dlp" >&2
    exit 1
fi

mkdir -p "$OUT"

echo "Source: $URL"
echo "Destination: $OUT"
echo

yt-dlp \
    --ignore-errors \
    --no-abort-on-error \
    --no-overwrites \
    --no-download \
    --write-subs \
    --write-auto-subs \
    --sub-langs 'en-orig,en,en-.*' \
    --sub-format 'vtt' \
    --download-archive "$ARCHIVE" \
    --sleep-requests 1 \
    --sleep-subtitles 2 \
    --output "$OUT/%(id)s.%(ext)s" \
    "$URL"

echo
echo "Download completed."

echo
echo "Subtitle files:"
find "$OUT" -maxdepth 1 -type f -name '*.vtt' | wc -l

echo
echo "Distinct video IDs:"
find "$OUT" -maxdepth 1 -type f -name '*.vtt' \
    -printf '%f\n' \
    | sed -E 's/\.en(-orig)?([.-][^.]+)*\.vtt$//' \
    | sort -u \
    | wc -l
