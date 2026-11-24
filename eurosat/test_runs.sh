#!/bin/bash
set -e
for seed in 1 123 16 20 58; do
    python test.py --seed "$seed" "$@"
done
