#!/usr/bin/env bash
# Download BAAI/bge-m3 into models/bge-m3 with curl resume (-C -), then run
# the real indexing. Robust against the stalling HF connections observed on
# this machine: every retry continues at the exact byte where the last died.
set -u
DIR="models/bge-m3"
mkdir -p "$DIR/1_Pooling"

BASE="https://huggingface.co/BAAI/bge-m3/resolve/main"

fetch_small() {
  curl -sSL --retry 5 --max-time 120 -o "$DIR/$1" "$BASE/$1"
}

echo "== small files =="
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
         sentencepiece.bpe.model modules.json config_sentence_transformers.json \
         1_Pooling/config.json; do
  [ -s "$DIR/$f" ] || { fetch_small "$f" && echo "  ok $f"; }
done

echo "== weights (pytorch_model.bin, 2.27 GB, resume per retry) =="
for i in $(seq 1 120); do
  echo "--- weights attempt $i ($(du -m "$DIR/pytorch_model.bin" 2>/dev/null | cut -f1) MB so far)"
  curl -L -C - --max-time 600 --retry 0 -o "$DIR/pytorch_model.bin" \
       "$BASE/pytorch_model.bin" && break
  sleep 5
done

SIZE=$(du -m "$DIR/pytorch_model.bin" | cut -f1)
echo "== weights done: $SIZE MB =="
if [ "$SIZE" -lt 2000 ]; then
  echo "model file suspiciously small; aborting"
  exit 1
fi

echo "== indexing with local bge-m3 =="
export NEXUS_EMBEDDING_MODEL_PATH="$DIR"
python scripts/index_embeddings.py --real-content --from-api --ingest-tree
