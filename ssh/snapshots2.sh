#!/bin/bash
# GoML Snapshot Script
# Usage: ./snapshot.sh [project_dir] [output_file]
# Example: ./snapshot.sh ~/goml snapshot_v4.txt

PROJECT_DIR="${1:-.}"
VERSION="${2:-snapshot_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${VERSION}.txt"

cd "$PROJECT_DIR" || { echo "Error: cannot cd to $PROJECT_DIR"; exit 1; }

echo "# GoML v3 Snapshot - $(date)" > "$OUTPUT"
echo "" >> "$OUTPUT"

# Project structure
echo "## Project Structure" >> "$OUTPUT"
echo '```' >> "$OUTPUT"
find . -type f -name "*.go" -o -name "go.mod" -o -name "go.sum" | \
  grep -v vendor | sort | sed 's|^\./||' | \
  while IFS= read -r f; do echo "$f"; done | \
  tree --fromfile --noreport 2>/dev/null || \
  find . -type f \( -name "*.go" -o -name "go.mod" \) | grep -v vendor | sort | sed 's|^\./|  |'
echo '```' >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Line counts
echo "## Line Counts" >> "$OUTPUT"
echo '```' >> "$OUTPUT"
find . -type f -name "*.go" | grep -v vendor | sort | while IFS= read -r f; do
  wc -l < "$f" | tr -d ' ' | xargs -I{} echo "{} $f"
done >> "$OUTPUT"
echo "---" >> "$OUTPUT"
find . -type f -name "*.go" | grep -v vendor | xargs cat | wc -l | xargs -I{} echo "{} TOTAL" >> "$OUTPUT"
echo '```' >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Each file
find . -type f \( -name "*.go" -o -name "go.mod" \) | grep -v vendor | sort | while IFS= read -r f; do
  echo "## File: $f" >> "$OUTPUT"
  echo '```go' >> "$OUTPUT"
  cat "$f" >> "$OUTPUT"
  echo "" >> "$OUTPUT"
  echo '```' >> "$OUTPUT"
  echo "" >> "$OUTPUT"
done

LINES=$(wc -l < "$OUTPUT")
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "✅ Snapshot saved: $OUTPUT ($LINES lines, $SIZE)"