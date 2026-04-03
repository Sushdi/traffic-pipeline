#!/usr/bin/env bash
# One-time project bootstrap
set -e

echo "=== 1. Creating directory structure ==="
mkdir -p data/raw data/processed models metrics \
         src/api monitoring .github/workflows
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep

echo "=== 2. Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== 3. Initialising Git + DVC ==="
git init
dvc init

echo "=== 4. Configuring DVC remote (Google Drive) ==="
echo "Paste your Google Drive folder ID (from the URL after /folders/):"
read -r GDRIVE_ID
dvc remote add -d myremote gdrive://"$GDRIVE_ID"
dvc remote modify myremote gdrive_acknowledge_abuse true

echo "=== 5. Committing initial setup ==="
git add .
git commit -m "init: project scaffold"

echo "=== 6. Pushing DVC config ==="
git add .dvc/config
git commit -m "config: dvc remote"

echo ""
echo "Done! Next steps:"
echo "  1. Add secrets to GitHub: TOMTOM_API_KEY, GH_PAT, GDRIVE_CREDENTIALS"
echo "  2. Add Actions variable COLLECTION_START_EPOCH = \$(date +%s)"
echo "  3. On Saturday midnight Munich time:"
echo "     → GitHub → Actions → 'Collect Traffic Data' → Run workflow"
echo ""
echo "After 3 days:"
echo "  dvc pull"
echo "  python src/preprocess.py"
echo "  python src/train.py"
echo "  docker-compose up"