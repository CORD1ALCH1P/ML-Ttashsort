#!/usr/bin/env bash
# Full pipeline: split data -> train all architectures -> evaluate -> compare.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
cd src
python3 prepare_data.py
python3 train.py --arch all --epochs-head 6 --epochs-finetune 6
python3 evaluate.py --arch all
python3 compare_models.py
