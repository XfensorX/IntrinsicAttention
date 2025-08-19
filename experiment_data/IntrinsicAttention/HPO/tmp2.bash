#!/usr/bin/env bash
# Usage: ./watch_returns.sh /path/to/root   (defaults to current dir)
set -u

DIR="${1:-.}"
INTERVAL=5        # seconds between updates
DURATION=120      # total runtime in seconds

declare -A prev   # previous values per file

end=$((SECONDS + DURATION))

# colors
reset="\e[0m"
clr_green="\e[32m"
clr_red="\e[31m"
clr_yellow="\e[33m"

while (( SECONDS < end )); do
  # get all result.json files fresh each round
  mapfile -t files < <(find "$DIR" -type f -name "result.json" 2>/dev/null | sort)

  # clear screen & header
  printf "\e[2J\e[H"
  printf "\e[1m%s\e[0m — scanning %s (every %ss; %ds left)\n\n" \
    "$(date)" "$DIR" "$INTERVAL" $((end-SECONDS))

  for f in "${files[@]}"; do
    # last occurrence of "episode_return_mean"
    line=$(grep -E '"episode_return_mean"' "$f" | tail -n 1)
    if [[ -z "$line" ]]; then
      printf "%s: (no episode_return_mean found)\n" "$f"
      continue
    fi

    # extract numeric value
    val=$(sed -E 's/.*"episode_return_mean"[[:space:]]*:[[:space:]]*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?).*/\1/' <<<"$line")
    if [[ -z "$val" ]]; then
      printf "%s: (could not parse number)\n" "$f"
      continue
    fi

    prev_val="${prev[$f]:-}"
    arrow=""; color=""

    if [[ -n "$prev_val" ]]; then
      # compare numerically
      cmp=$(awk -v a="$val" -v b="$prev_val" 'BEGIN{if (a>b) print "gt"; else if (a<b) print "lt"; else print "eq"}')
      case "$cmp" in
        gt) color="$clr_green"; arrow="↑";;
        lt) color="$clr_red";   arrow="↓";;
        eq) ;; # no color
      esac
    fi

    if [[ -n "$color" ]]; then
      printf "%s: %b%s%b %s\n" "$f" "$color" "$val" "$reset" "$arrow"
    else
      printf "%s: %s\n" "$f" "$val"
    fi

    # store current as previous for next loop
    prev["$f"]="$val"
  done

  (( SECONDS < end )) && sleep "$INTERVAL"
done
