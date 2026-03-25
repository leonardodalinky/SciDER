#!/bin/bash
# Create a clean orphan branch and force push to HF remote.
# This avoids binary file history issues with HF's xet requirement.
# Requires: pip install huggingface_hub[hf_xet]
set -e

# Ensure xet is configured
git xet install

# Step 1: Track binary extensions with xet
git xet track "*.gif"
git xet track "*.webp"
git xet track "*.png"
git xet track "*.jpg"
git xet track "*.jpeg"
git xet track "*.pkl"
git xet track "*.dat"
git xet track "*.csv"

# Step 2: Create orphan branch with a single clean commit
git checkout --orphan hf_deploy
git add -A
GIT_AUTHOR_NAME="leonardklin" \
GIT_AUTHOR_EMAIL="leonard.keilin@gmail.com" \
GIT_COMMITTER_NAME="leonardklin" \
GIT_COMMITTER_EMAIL="leonard.keilin@gmail.com" \
git commit -m "deploy: SciDER to HuggingFace Space"

# Step 3: Force push as main to HF
git push hf hf_deploy:main --force

# Step 4: Go back to hf_space and delete temp branch
git checkout hf_space
git branch -D hf_deploy

echo "Done. HF remote main is now a clean single-commit deploy."
