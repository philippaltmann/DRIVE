#!/usr/bin/env bash

ENVS=(
  "Matrix-IPD"
  "CoinGame-2"
  "CoinGame-4"
  "Harvest-12"
)

DRIFTS=(
  "identity"
  "linear"
  "exponential_decay"
  "stepwise_increase"
  "cos_damped"
  "noisy"
)

ALGS=(
  "DRIVE"
  "MATE"
  "MEDIATE"
  "LToS"
  "LIO"
  "IAC"
  "IA"
)

# Optional seed argument
SEED="${1:-}"

# Timestamp for unique logfiles
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for ENV in "${ENVS[@]}"; do
  for ALG in "${ALGS[@]}"; do
    for DRIFT in "${DRIFTS[@]}"; do

      if [[ -n "$SEED" ]]; then
        python -m train "$ALG" "$ENV" -d "$DRIFT" -s "$SEED" \
          > "./_logfiles/${ENV}_${DRIFT}_${ALG}_${SEED}.out" &
      else
        python -m train "$ALG" "$ENV" -d "$DRIFT" \
          > "./_logfiles/${ENV}_${DRIFT}_${ALG}_${TIMESTAMP}.out" &
      fi

    done
  done
done