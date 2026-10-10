#!/bin/sh
set -e
cd "$(dirname "$0")/.."
python3 -m pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean --onedir --windowed --name "iGEM Motion Studio" --paths src src/merged_abm_desktop_app.py
