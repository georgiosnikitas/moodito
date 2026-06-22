#!/usr/bin/env bash
# Build the standalone Moodito.app bundle.
#
# Usage:  ./build.sh
# Output: dist/Moodito.app
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --only-binary :all: --upgrade pip >/dev/null
pip install --require-hashes --only-binary :all: --no-binary rumps -r requirements.txt >/dev/null
pip install --only-binary :all: pyinstaller >/dev/null

python -m PyInstaller --noconfirm --clean moodito.spec

echo
echo "Built dist/Moodito.app"
echo "Run it with:  open dist/Moodito.app"
echo "Install it with:  cp -R dist/Moodito.app /Applications/"
