"""
Search-R1 Level 2 多模态 GRPO 训练用的 reward 函数。

输入约定（由 scripts/data_process/coco_image_qa.py 生成的 ground_truth）：
  ground_truth = {
      "keywords": ["dog", "frisbee", ...],   # 从 caption 抽出的关键词
      "captions": ["A dog catches a frisbee.", ...],  # 该图的全部 caption
      "image_id": 12345,
      "file_name": "000000012345.jpg",
  }

打分规则（简单可解释，便于训练初期收敛信号清晰）：
  - 强制格式：response 必须含 <answer>...</answer>，否则 0 分
  - 关键词命中：<answer> 中命中的关键词比例（lexical recall）作为主分数 [0,1]
  - 额外 tool-use 奖励：response 中至少出现一次 <search>...</search> 时加 0.2
  - 上限封顶到 score（默认 1.0），下限 0.0
"""

import random
import re
import string
from typing import List


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEARCH_PATTERN = re.compile(r"<search>.*?</search>", re.DOTALL)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def extract_answer(solution_str: str) -> str:
    """从模型输出里抽取最后一个 <answer>...</answer>。"""
    matches = _ANSWER_PATTERN.findall(solution_str)
    if not matches:
        return ""
    return matches[-1].strip()


def _to_str_list(value) -> List[str]:
    """
    将 ground_truth 字段安全地规整为 List[str]。
    parquet 反序列化出的字段可能是 numpy.ndarray / pandas Series / list / None / 标量字符串，
    统一转换避免 `if not value` 这类布尔判断在 ndarray 上触发 ValueError。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        items = list(value)
    except TypeError:
        return [str(value)]
    return [str(x) for x in items if x is not None]


def keyword_recall(answer: str, keywords: List[str]) -> float:
    if len(keywords) == 0:
        return 0.0
    norm_answer = _normalize(answer)
    answer_tokens = set(norm_answer.split())
    if len(answer_tokens) == 0:
        return 0.0
    hits = 0
    for kw in keywords:
        kw_norm = _normalize(kw)
        if len(kw_norm) == 0:
            continue
        # 多词 keyword 视为整体短语，单词 keyword 走 token 命中
        if " " in kw_norm:
            if kw_norm in norm_answer:
                hits += 1
        else:
            if kw_norm in answer_tokens:
                hits += 1
    return hits / len(keywords)


def compute_score_em(
    solution_str: str,
    ground_truth: dict,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """
    与 verl 现有 qa_em.compute_score_em 同名同签名，便于直接被 main_ppo._select_rm_score_fn 路由。
    """
    if not isinstance(ground_truth, dict):
        return 0.0

    answer = extract_answer(solution_str)
    do_print = random.randint(1, 64) == 1

    keywords = _to_str_list(ground_truth.get("keywords"))
    captions = _to_str_list(ground_truth.get("captions"))

    if do_print:
        print("--------------------------------")
        print(f"[coco_image_qa] keywords: {keywords}")
        print(f"[coco_image_qa] captions: {captions[:2]}")
        print(f"[coco_image_qa] extracted answer: {answer!r}")
        print(f"[coco_image_qa] solution tail: ...{solution_str[-300:]}")

    if len(answer) == 0:
        return format_score

    recall = keyword_recall(answer, keywords)
    used_search = bool(_SEARCH_PATTERN.search(solution_str))

    final = recall + (0.2 if used_search else 0.0)
    if final > score:
        final = score
    if final < 0.0:
        final = 0.0
    return final
