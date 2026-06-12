"""
基于 COCO val2017 caption 数据，构造一个 *最小可跑通* 的图像检索 QA 数据集，
用于 Search-R1 Level 2 的简化版 GRPO 训练。

构造思路（极简版，不追求高质量监督信号）：
  - 把每张图片的 caption 作为问题的"答案锚点"
  - 让模型在不直接看到原图的情况下，回答"找一张包含 X 的图，并描述其中的 Y"
  - reward 用规则评估：模型的 <answer> 是否包含 caption 关键词（lexical EM/F1）

输出 parquet 字段对齐原 nq_search.py 的格式，便于直接喂给 verl 训练管线：
  data_source / prompt / ability / reward_model / extra_info
"""

import argparse
import json
import os
import random
import re
from typing import List

import pandas as pd


SYSTEM_INSTRUCTION = (
    "You are a multimodal reasoning assistant. Answer the user's question using this protocol.\n"
    "1. Reason inside <think> and </think>.\n"
    "2. When you need visual evidence, write <search> a short english phrase describing the image you want </search>. "
    "The system will return the top-k matching images wrapped in <information> ... </information>.\n"
    "3. Inspect the returned images, then provide the final answer inside <answer> and </answer>."
)


def load_coco_captions(annotation_path: str) -> List[dict]:
    """加载 COCO captions_val2017.json 标注，返回 [{image_id, file_name, captions:[...]}, ...]。"""
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    image_id_to_meta = {img["id"]: img for img in ann["images"]}
    image_id_to_captions: dict = {}
    for cap in ann["annotations"]:
        image_id_to_captions.setdefault(cap["image_id"], []).append(cap["caption"].strip())

    items = []
    for image_id, caps in image_id_to_captions.items():
        meta = image_id_to_meta.get(image_id)
        if not meta:
            continue
        items.append({
            "image_id": image_id,
            "file_name": meta["file_name"],
            "captions": caps,
        })
    return items


def load_captions_from_parquet(parquet_path: str) -> List[dict]:
    """
    从 modelscope `lmms-lab/COCO-Caption` 这类 parquet 直接读 captions。
    parquet schema 期望含字段：file_name, id, answer(List[str])。
    返回与 load_coco_captions 相同的结构。
    """
    df = pd.read_parquet(parquet_path)
    items: List[dict] = []
    for row_idx, row in df.iterrows():
        captions_field = row.get("answer")
        if captions_field is None:
            continue
        captions = [str(c).strip() for c in list(captions_field) if str(c).strip()]
        if not captions:
            continue

        try:
            image_id = int(row["id"])
        except (KeyError, ValueError, TypeError):
            image_id = int(row_idx)

        file_name = str(row.get("file_name") or f"img_{image_id:08d}.jpg")
        items.append({
            "image_id": image_id,
            "file_name": file_name,
            "captions": captions,
        })
    return items


# 从 caption 里抽关键 token 作为答案锚点（粗略版，不引入 nltk 依赖）
# 只过滤纯虚词，保留所有名词性 token（包括 person/man/woman/people 这些 COCO 高频核心词）
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "and", "or", "with", "for", "by", "this", "that", "these", "those", "as",
    "it", "its", "be", "has", "have", "had", "from", "into", "his", "her", "their",
    "some", "very", "near", "next", "there", "here", "they", "them", "him",
    "but", "not", "than", "then", "while", "when", "where", "what", "which", "who",
}


def extract_keywords(caption: str, max_n: int = 4) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", caption.lower())
    keywords = [t for t in tokens if t not in _STOPWORDS]
    # 去重，保持顺序
    seen = set()
    out = []
    for t in keywords:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def build_question_from_caption(caption: str) -> str:
    """把 caption 改写成"先搜图，再描述"的问题。"""
    # 截断过长的 caption，避免把答案直接写进 question
    short = caption.split(".")[0].strip()
    if len(short) > 100:
        short = short[:100]
    return (
        f"Find an image that matches the following description, then describe in one short sentence "
        f"what the main subject is doing. Description: \"{short}\"."
    )


def make_sample(item: dict, idx: int, split: str) -> dict:
    caption = random.choice(item["captions"])
    question = build_question_from_caption(caption)
    keywords = extract_keywords(caption)

    user_content = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question: {question}"
    )

    return {
        "data_source": "coco_image_qa",
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "multimodal-search-qa",
        "reward_model": {
            "style": "rule",
            # ground_truth 既给出关键词列表用于 lexical 评估，也保留原始 caption 备查
            "ground_truth": {
                "keywords": keywords,
                "captions": item["captions"],
                "image_id": item["image_id"],
                "file_name": item["file_name"],
            },
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "image_id": item["image_id"],
            "file_name": item["file_name"],
        },
    }


def main():
    parser = argparse.ArgumentParser("Build COCO-based multimodal search QA dataset")
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="COCO captions json，例如 .../annotations/captions_val2017.json")
    parser.add_argument("--parquet_path", type=str, default=None,
                        help="modelscope lmms-lab/COCO-Caption 这类 parquet 路径（与 annotation_path 二选一）")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.annotation_path and not args.parquet_path:
        raise SystemExit("必须指定 --annotation_path 或 --parquet_path 之一")

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    if args.parquet_path:
        print(f"[1/3] loading captions from parquet {args.parquet_path}")
        items = load_captions_from_parquet(args.parquet_path)
    else:
        print(f"[1/3] loading captions from {args.annotation_path}")
        items = load_coco_captions(args.annotation_path)
    print(f"  total images with captions: {len(items)}")

    random.shuffle(items)
    train_items = items[: args.train_size]
    test_items = items[args.train_size : args.train_size + args.test_size]
    print(f"  train={len(train_items)} test={len(test_items)}")

    print("[2/3] building train samples")
    train_rows = [make_sample(it, i, "train") for i, it in enumerate(train_items)]
    print("[2/3] building test samples")
    test_rows = [make_sample(it, i, "test") for i, it in enumerate(test_items)]

    print("[3/3] writing parquet")
    train_path = os.path.join(args.out_dir, "train.parquet")
    test_path = os.path.join(args.out_dir, "test.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(test_rows).to_parquet(test_path)
    print(f"  wrote {train_path} and {test_path}")


if __name__ == "__main__":
    main()
