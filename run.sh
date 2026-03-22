#!/usr/bin/env bash
set -euo pipefail

LOG="compare_results.txt"

LIBS="${CONDA_PREFIX:-}/lib:/home/slava/.sdkman/candidates/java/21.0.5-tem/lib/server"
export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Отключим прогресс-бары глобально на всякий случай
export TQDM_DISABLE=1

for L in 5000 10000 20000 50000 100000 200000; do
  {
    echo "=============================="
    echo "Run started: $(date -Iseconds)"
    echo "Command: python -m spar_k_means_bert.run --use-cache -d msmarco -l $L"
  } >>"$LOG"

  # Только stdout уходит в лог;stderr (в т.ч. tqdm) останется в терминале
  python -m spar_k_means_bert.run --use-cache -d msmarco -l "$L" >>"$LOG"

  if [ -d runs/spar_k_means_bert/lucene_inverted_index ]; then
    du -h -d 1 runs/spar_k_means_bert/lucene_inverted_index >>"$LOG"
  else
    echo "lucene_inverted_index not found; skipping du." >>"$LOG"
  fi

  echo "Cleaning up lucene_inverted_index..." >>"$LOG"
  rm -rf runs/spar_k_means_bert/lucene_inverted_index
  echo >>"$LOG"
done
