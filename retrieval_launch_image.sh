#!/usr/bin/env bash
# 启动 CLIP-based 图像检索服务（Search-R1 Level 2）。
# 监听 0.0.0.0:8000，POST /retrieve 接口与原文本版兼容。

set -e

INDEX_DIR=${INDEX_DIR:-/var/lib/container/dataset/yxqiu/datasets/coco_index}
CLIP_MODEL=${CLIP_MODEL:-/var/lib/container/dataset/yxqiu/models/clip-vit-base-patch32}
IMAGE_DIR=${IMAGE_DIR:-/var/lib/container/dataset/yxqiu/datasets/coco/val2017}

if [ ! -f "$INDEX_DIR/clip_image.index" ]; then
    echo "[ERROR] $INDEX_DIR/clip_image.index not found."
    echo "Run build_image_index.py first, e.g.:"
    echo "  python search_r1/search/build_image_index.py \\"
    echo "    --image_dir $IMAGE_DIR \\"
    echo "    --annotation_path /var/lib/container/dataset/yxqiu/datasets/coco/annotations/captions_val2017.json \\"
    echo "    --clip_model_path $CLIP_MODEL \\"
    echo "    --out_dir $INDEX_DIR --use_fp16"
    exit 1
fi

python3 search_r1/search/image_retrieval_server.py \
    --index_path $INDEX_DIR/clip_image.index \
    --corpus_path $INDEX_DIR/image_corpus.jsonl \
    --clip_model_path $CLIP_MODEL \
    --topk 3 \
    --port 8000 \
    --device cuda:0 \
    --use_fp16
