#!/usr/bin/env bash
# Launch the Streamlit web app for interactive waste-classification testing.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
streamlit run webapp/app.py --server.address=0.0.0.0 --server.port=8501
