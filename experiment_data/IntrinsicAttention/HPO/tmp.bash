#!/usr/bin/env bash
echo "File,Value,Current Epochs"
find . -name "result.json" -type f -print0 | while IFS= read -r -d '' file; do
  value=$(grep -E '"episode_return_mean"' "$file" \
          | tail -n 1 \
          | sed -E 's/.*"episode_return_mean": *(-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?).*/\1/')
  # strip any trailing comma just in case
  value=${value%,}
  epochs=$(wc -l < "$file")
  printf '%s,%s,%s\n' "$file" "$value" "$epochs"
done
