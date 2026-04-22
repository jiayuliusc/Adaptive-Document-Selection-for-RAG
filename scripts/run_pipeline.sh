#!/usr/bin/env bash

set -euo pipefail

SAMPLE_SIZE="${1:-3000}"
NEGATIVES_PER_QUESTION="${2:-1}"

python experiment.py \
  --sample-size "$SAMPLE_SIZE" \
  --negatives-per-question "$NEGATIVES_PER_QUESTION"