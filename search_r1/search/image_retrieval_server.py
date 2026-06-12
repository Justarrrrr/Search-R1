"""
CLIP-based 文本→图像 检索服务（Search-R1 Level 2 多模态版本）。

兼容原 retrieval_server.py 的 HTTP 接口约定：
  POST /retrieve  body={"queries":[...], "topk":3, "return_scores":true}
  返回 {"result": [[{"document": {...}, "score": float}, ...], ...]}

与文本版本的关键区别：
- 检索器使用 CLIP 文本编码器，对 query 做 embedding
- 索引内容是图像 embedding（事先用 CLIP 视觉编码器对图像库离线计算并存入 FAISS）
- 返回的 document 内含 image_id / file_name / image_url / caption 等字段，
  方便客户端把图像注入回 MLLM 的 prompt
"""

import argparse
import json
import os
from typing import List, Optional

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor


def load_corpus_jsonl(corpus_path: str) -> List[dict]:
    """加载图像 corpus（每行一个 dict，至少含 id/file_name 字段，可附带 caption）。"""
    items: List[dict] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class ClipTextEncoder:
    """只跑 CLIP 文本侧的编码器，复用其文本塔做 query embedding。"""

    def __init__(self, model_path: str, device: str, use_fp16: bool):
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_path, use_fast=True)
        model = CLIPModel.from_pretrained(model_path)
        model.eval()
        model.to(self.device)
        if use_fp16 and self.device.type == "cuda":
            model = model.half()
        self.model = model
        self.use_fp16 = use_fp16

    @torch.no_grad()
    def encode(self, queries: List[str]) -> np.ndarray:
        if isinstance(queries, str):
            queries = [queries]
        inputs = self.processor(text=queries, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # transformers>=5.x 返回 BaseModelOutputWithPooling，其 pooler_output 是 projection 之后的 text embedding
        outputs = self.model.get_text_features(**inputs)
        text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        # 归一化以便用内积当余弦相似度
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return text_features.detach().float().cpu().numpy().astype(np.float32, order="C")


class ImageRetriever:
    """简单的稠密图像检索器：CLIP 文本 query → FAISS top-k → 返回图像元信息。"""

    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        clip_model_path: str,
        topk: int,
        device: str,
        use_fp16: bool,
        image_url_prefix: str,
    ):
        self.topk = topk
        self.image_url_prefix = image_url_prefix.rstrip("/")
        self.encoder = ClipTextEncoder(clip_model_path, device=device, use_fp16=use_fp16)

        self.corpus = load_corpus_jsonl(corpus_path)
        self.index = faiss.read_index(index_path)

        if self.index.ntotal != len(self.corpus):
            raise RuntimeError(
                f"Index size ({self.index.ntotal}) != corpus size ({len(self.corpus)}); "
                "请确认 index 与 corpus 是同一次构建产生的"
            )

    def _format_doc(self, item: dict) -> dict:
        """把 corpus 条目转成对外暴露的 document 结构。"""
        file_name = item.get("file_name", "")
        image_path = item.get("image_path", "")
        return {
            "id": item.get("id"),
            "file_name": file_name,
            "image_path": image_path,
            "image_url": f"{self.image_url_prefix}/{file_name}" if file_name else "",
            "caption": item.get("caption", ""),
            # 兼容原文本版返回字段，便于复用客户端代码
            "contents": item.get("caption", ""),
        }

    def batch_search(self, queries: List[str], topk: int, return_scores: bool):
        if not queries:
            return [], []
        query_emb = self.encoder.encode(queries)
        scores, idxs = self.index.search(query_emb, k=topk)

        results = []
        score_lists = []
        for row_idx in range(len(queries)):
            row_results = []
            row_scores = []
            for rank in range(topk):
                doc_idx = int(idxs[row_idx, rank])
                if doc_idx < 0:
                    continue
                row_results.append(self._format_doc(self.corpus[doc_idx]))
                row_scores.append(float(scores[row_idx, rank]))
            results.append(row_results)
            score_lists.append(row_scores)
        return results, score_lists


# -----------------------------
# FastAPI server
# -----------------------------

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI(title="Search-R1 Image Retrieval Server")
retriever: Optional[ImageRetriever] = None


@app.get("/health")
def health():
    return {"status": "ok", "corpus_size": retriever.index.ntotal if retriever else 0}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or retriever.topk
    results, scores = retriever.batch_search(request.queries, topk=topk, return_scores=request.return_scores)

    resp = []
    for row_idx, row in enumerate(results):
        if request.return_scores:
            combined = [{"document": doc, "score": s} for doc, s in zip(row, scores[row_idx])]
            resp.append(combined)
        else:
            resp.append(row)
    return {"result": resp}


def main():
    parser = argparse.ArgumentParser("CLIP-based image retrieval server for Search-R1 Level 2")
    parser.add_argument("--index_path", type=str, required=True, help="FAISS index 路径（图像 embedding）")
    parser.add_argument("--corpus_path", type=str, required=True, help="图像 corpus jsonl，与 index 行序对齐")
    parser.add_argument("--clip_model_path", type=str, required=True, help="CLIP 模型本地路径")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument(
        "--image_url_prefix",
        type=str,
        default="",
        help="可选：图像服务的 URL 前缀，便于客户端通过 HTTP 直接拿到图片，留空则只返回本地路径",
    )

    args = parser.parse_args()

    global retriever
    retriever = ImageRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        clip_model_path=args.clip_model_path,
        topk=args.topk,
        device=args.device,
        use_fp16=args.use_fp16,
        image_url_prefix=args.image_url_prefix,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
