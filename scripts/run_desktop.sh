#!/bin/sh
set -e
cd "$(dirname "$0")/.."
python3 src/merged_abm_desktop_app.py
