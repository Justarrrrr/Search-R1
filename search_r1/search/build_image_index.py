"""
基于 CLIP 视觉编码器，对图像库做离线 embedding 并构建 FAISS 索引。

支持两种数据源：
  1) image_dir + COCO captions json（原始 COCO 解压方式）
  2) parquet_path（modelscope 的 lmms-lab/COCO-Caption 这类已打包好 image bytes + caption 的 parquet）

输出：
  - <out_dir>/image_corpus.jsonl  每行一个图像条目
  - <out_dir>/clip_image.index    FAISS Flat 索引（IndexFlatIP，需要归一化向量）
  - 仅 parquet 模式：<out_dir>/images/  解出来的 jpg，供 retriever 通过 image_path 服务
约定：image_corpus.jsonl 的行序与 FAISS 索引的向量顺序严格一致。
"""

import argparse
import io
import json
import os
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_coco_captions(annotation_path: str) -> dict:
    """读取 COCO captions_val2017.json，返回 {image_id: [caption,...]}。"""
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    image_id_to_captions: dict = {}
    for cap in ann.get("annotations", []):
        image_id_to_captions.setdefault(cap["image_id"], []).append(cap["caption"].strip())
    return image_id_to_captions


def scan_image_files(image_dir: str, limit: Optional[int]) -> List[str]:
    files = sorted(os.listdir(image_dir))
    files = [name for name in files if name.lower().endswith((".jpg", ".jpeg", ".png"))]
    if limit:
        files = files[:limit]
    return files


def filter_openable_images(image_dir: str, file_names: List[str]) -> List[str]:
    """预扫一遍：丢弃任何 PIL 打不开的图，保证后续 encode 阶段不会出现零向量。"""
    kept: List[str] = []
    dropped: int = 0
    for name in tqdm(file_names, desc="verify images"):
        path = os.path.join(image_dir, name)
        try:
            with Image.open(path) as img:
                img.verify()  # 仅做完整性校验，不解码全图
            kept.append(name)
        except Exception as e:
            dropped += 1
            print(f"[WARN] drop unreadable image {path}: {e}")
    print(f"  verified {len(kept)} images, dropped {dropped}")
    return kept


def build_corpus_from_coco(image_dir: str, annotation_path: Optional[str], file_names: List[str]) -> List[dict]:
    """根据已校验过的文件列表构建 corpus（与 index 行序严格对齐）。"""
    captions = load_coco_captions(annotation_path) if annotation_path else {}

    corpus: List[dict] = []
    for file_name in file_names:
        # COCO 文件名形如 000000000139.jpg
        stem, _ = os.path.splitext(file_name)
        try:
            image_id = int(stem)
        except ValueError:
            image_id = stem

        caption_list = captions.get(image_id, [])
        primary_caption = caption_list[0] if caption_list else ""

        corpus.append({
            "id": image_id,
            "file_name": file_name,
            "image_path": os.path.join(image_dir, file_name),
            "caption": primary_caption,
            "all_captions": caption_list,
        })
    return corpus


def build_corpus_from_parquet(parquet_path: str, images_out_dir: str, limit: Optional[int]) -> List[dict]:
    """
    从 modelscope `lmms-lab/COCO-Caption` 这类 parquet 读取 image bytes + captions，
    把图像解码后落盘到 images_out_dir，并返回与 jsonl 行序对齐的 corpus。
    parquet schema 期望包含字段：image{bytes,path}, file_name, id, answer(List[str]), coco_url（可选）。
    """
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.iloc[:limit]

    os.makedirs(images_out_dir, exist_ok=True)
    corpus: List[dict] = []
    dropped = 0

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="extract images"):
        image_field = row["image"]
        # parquet 反序列化后，struct 字段会变成 dict
        if not isinstance(image_field, dict) or "bytes" not in image_field:
            dropped += 1
            continue
        img_bytes = image_field["bytes"]
        if not img_bytes:
            dropped += 1
            continue

        # 校验图像可解码
        try:
            with Image.open(io.BytesIO(img_bytes)) as im:
                im.verify()
        except Exception as e:
            print(f"[WARN] drop unreadable parquet row {row_idx}: {e}")
            dropped += 1
            continue

        file_name = str(row.get("file_name") or image_field.get("path") or f"img_{row_idx:08d}.jpg")
        out_path = os.path.join(images_out_dir, file_name)
        # 写图：原始字节透写，避免 PIL 重编码丢质量
        if not os.path.exists(out_path):
            with open(out_path, "wb") as f:
                f.write(img_bytes)

        captions = row.get("answer")
        if captions is None:
            caption_list: List[str] = []
        else:
            caption_list = [str(c) for c in list(captions)]

        try:
            image_id = int(row["id"])
        except (KeyError, ValueError, TypeError):
            image_id = row_idx

        corpus.append({
            "id": image_id,
            "file_name": file_name,
            "image_path": out_path,
            "caption": caption_list[0] if caption_list else "",
            "all_captions": caption_list,
        })

    print(f"  parquet rows: {len(df)}, kept: {len(corpus)}, dropped: {dropped}")
    return corpus


@torch.no_grad()
def encode_images(corpus: List[dict], clip_model_path: str, device: str, batch_size: int, use_fp16: bool) -> np.ndarray:
    """
    逐 batch 跑 CLIP 视觉编码，返回 (N, D) 的归一化 embedding 矩阵。
    前置条件：corpus 中每个 image_path 都已被校验过可解码，
    本函数不做容错——任何打开失败都视为错误并直接抛出，避免污染索引。
    """
    processor = CLIPProcessor.from_pretrained(clip_model_path, use_fast=True)
    model = CLIPModel.from_pretrained(clip_model_path)
    model.eval()
    model.to(device)
    if use_fp16 and device.startswith("cuda"):
        model = model.half()

    all_embeddings: List[np.ndarray] = []

    for start in tqdm(range(0, len(corpus), batch_size), desc="encoding images"):
        batch = corpus[start:start + batch_size]
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if use_fp16 and device.startswith("cuda") and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # transformers>=5.x 返回 BaseModelOutputWithPooling，其 pooler_output 已是 projection 之后的 image embedding
        outputs = model.get_image_features(**inputs)
        feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        feats = torch.nn.functional.normalize(feats, dim=-1)
        feats_np = feats.detach().float().cpu().numpy().astype(np.float32, order="C")
        all_embeddings.append(feats_np)

        for img in images:
            img.close()

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser("Build CLIP image index for Search-R1 Level 2")
    # 数据源二选一
    parser.add_argument("--parquet_path", type=str, default=None,
                        help="modelscope lmms-lab/COCO-Caption 这类含 image bytes 的 parquet 路径（推荐）")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="图像目录，例如 .../coco/val2017（与 parquet_path 二选一）")
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="COCO captions json（仅 image_dir 模式下使用）")

    parser.add_argument("--clip_model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张图，用于快速 smoke test")
    args = parser.parse_args()

    if not args.parquet_path and not args.image_dir:
        raise SystemExit("必须指定 --parquet_path 或 --image_dir 之一")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.parquet_path:
        print(f"[1/3] reading parquet {args.parquet_path}")
        images_out_dir = os.path.join(args.out_dir, "images")
        corpus = build_corpus_from_parquet(args.parquet_path, images_out_dir, args.limit)
    else:
        print(f"[1/3] scanning images in {args.image_dir}")
        raw_files = scan_image_files(args.image_dir, args.limit)
        print(f"  found {len(raw_files)} candidate files")
        valid_files = filter_openable_images(args.image_dir, raw_files)
        if not valid_files:
            raise RuntimeError(f"No openable images found under {args.image_dir}")
        corpus = build_corpus_from_coco(args.image_dir, args.annotation_path, valid_files)

    if not corpus:
        raise RuntimeError("empty corpus")
    print(f"  collected {len(corpus)} valid images")

    corpus_path = os.path.join(args.out_dir, "image_corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  corpus written -> {corpus_path}")

    print(f"[2/3] encoding with CLIP ({args.clip_model_path})")
    embeddings = encode_images(corpus, args.clip_model_path, args.device, args.batch_size, args.use_fp16)
    print(f"  embeddings shape: {embeddings.shape}")

    print("[3/3] building FAISS IndexFlatIP")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(args.out_dir, "clip_image.index")
    faiss.write_index(index, index_path)
    print(f"  index written -> {index_path} (ntotal={index.ntotal})")


if __name__ == "__main__":
    main()
