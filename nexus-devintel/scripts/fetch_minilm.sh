#!/usr/bin/env bash
# Download sentence-transformers/all-MiniLM-L6-v2 locally, then run the real
# indexing. Robust against stalling HF connections: every retry resumes with
# curl (-C -). ~90 MB total (vs ~2.3 GB for bge-m3).
set -u
DIR="models/minilm-l6-v2"
mkdir -p "$DIR"

BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

fetch_small() {
  curl -sSL --retry 5 --max-time 120 -o "$DIR/$1" "$BASE/$1"
}

echo "== small files =="
for f in config.json tokenizer.json tokenizer_config.json vocab.txt \
         special_tokens_map.json modules.json config_sentence_transformers.json \
         README.md; do
  [ -s "$DIR/$f" ] || { fetch_small "$f" && echo "  ok $f"; }
done

echo "== weights (pytorch_model.bin + model.safetensors fallback, resume per retry) =="
# Newer revisions ship model.safetensors; older ones pytorch_model.bin.
# Try safetensors first (smaller, faster to load), fall back to the bin.
for weights in model.safetensors pytorch_model.bin; do
  for i in $(seq 1 30); do
    echo "--- $weights attempt $i ($(du -m "$DIR/$weights" 2>/dev/null | cut -f1) MB so far)"
    curl -L -C - --max-time 600 --retry 0 -o "$DIR/$weights" \
         "$BASE/$weights" && break
    sleep 5
  done
  [ -s "$DIR/$weights" ] && break
done

echo "== indexing with local MiniLM =="
export NEXUS_EMBEDDING_MODEL_PATH="$DIR"
python scripts/index_embeddings.py

