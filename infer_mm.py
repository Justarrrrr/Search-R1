"""
Search-R1 Level 2 多模态推理链路（基于 Qwen3-VL-4B-Instruct + CLIP 图像检索服务）。

协议（与原 infer.py 对齐）：
  <think>...</think>
  <search> text query </search>           ← 文本 query 触发图像检索
  <information> ... </information>        ← 服务端返回 top-k 图像 + caption（注入下一轮）
  <answer> ... </answer>

实现要点（区别于原版纯文本 infer.py）：
- 使用 Qwen3-VL 的 chat template 原生 `tool` role 注入检索结果，
  让 chat-template 自己渲染 `<|vision_start|><|image_pad|><|vision_end|>`。
  绝不手写图像占位符或硬拼字符串。
- 多轮通过维护 messages 列表实现，每一轮把 assistant 的部分输出塞回 history，
  再把检索结果作为新一条 tool message 注入。
- 双 stop token：在 </search> 处停下让外层调检索，在 <|im_end|> 处自然结束。

使用方式：
  python infer_mm.py \
      --model_path /var/lib/container/dataset/yxqiu/models/Qwen3-VL-4B-Instruct \
      --retriever_url http://127.0.0.1:18000/retrieve \
      --question "Find an image of a cat sitting on a sofa, then tell me what color it is." \
      [--image path/to/user_image.jpg ...]

需要先启动 image_retrieval_server.py。
"""

import argparse
import re
from typing import List, Optional, Tuple

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList


USER_INSTRUCTION_TEMPLATE = (
    "Answer the given question. You have access to an image search engine.\n\n"
    "## Rules\n"
    "1. You MUST first reason inside <think> and </think>.\n"
    "2. If you need visual evidence, call the search engine: <search> your query </search>\n"
    "   The top-{topk} most relevant images will be returned between <information> and </information>.\n"
    "3. After receiving images, reason again inside <think> and </think>.\n"
    "4. You may search multiple times.\n"
    "5. When ready, give your final answer inside <answer> and </answer>.\n\n"
    "## Example\n"
    "Question: What color is the fire hydrant on the street?\n"
    "<think> I need to find an image of a fire hydrant on a street to determine its color. "
    "Let me search for relevant images. </think>\n"
    "<search> fire hydrant on the street </search>\n"
    "<information> [images of fire hydrants returned] </information>\n"
    "<think> Based on the returned images, I can see a red fire hydrant on the street. </think>\n"
    "<answer> The fire hydrant is red. </answer>\n\n"
    "Now answer the following question. Remember: you MUST start with <think> and use <search> to find images.\n\n"
    "Question: {question}"
)


SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


def extract_last_search_query(text: str) -> Optional[str]:
    matches = SEARCH_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


class StopOnTokens(StoppingCriteria):
    """支持多个停止序列：</search> 触发外层检索；<|im_end|>/<|endoftext|> 表示生成自然结束。"""

    def __init__(self, tokenizer, stop_strings: List[str]):
        self.target_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in stop_strings
        ]
        # 过滤掉空 / 非平凡的编码
        self.target_ids = [ids for ids in self.target_ids if len(ids) > 0]
        self.target_lengths = [len(ids) for ids in self.target_ids]
        self.min_len = min(self.target_lengths) if self.target_lengths else 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < self.min_len:
            return False
        for ids, length in zip(self.target_ids, self.target_lengths):
            tail = input_ids[0, -length:].tolist()
            if tail == ids:
                return True
        return False


def call_retriever(retriever_url: str, query: str, topk: int) -> List[dict]:
    """调用图像检索服务，返回 [{document:{...}, score:..}] 列表。"""
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(retriever_url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["result"][0]


# Qwen3-VL 图像占位符串：每张图一组 <|vision_start|><|image_pad|><|vision_end|>
# 与 processor.apply_chat_template 渲染 image content 时产生的字符串严格对齐。
QWEN_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def build_information_text(retrieved: List[dict]) -> Tuple[str, List[Image.Image]]:
    """
    把检索结果格式化成可直接拼到 prompt 末尾的 <information>...</information> 字符串，
    其中每张检索图都用 Qwen3-VL 的图像占位符序列表示。
    返回 (information 字符串, 实际成功打开的图像列表)。
    """
    images: List[Image.Image] = []
    parts: List[str] = ["\n<information>\n"]

    for idx, item in enumerate(retrieved, start=1):
        doc = item["document"] if "document" in item else item
        caption = (doc.get("caption") or "").strip().replace("\n", " ")
        image_path = doc.get("image_path", "")
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] cannot open retrieved image {image_path}: {e}")
            continue
        images.append(img)
        parts.append(f"Image {idx} (caption: \"{caption}\"): {QWEN_IMAGE_PLACEHOLDER}\n")

    parts.append("</information>\n")
    return "".join(parts), images


def truncate_after_search_tag(text: str) -> str:
    """
    把生成文本截断到第一个 </search> 之后（含 </search>），
    丢弃 stopping criteria 触发后模型可能多吐的尾巴文本，保持轮次干净。
    若不含 </search>，原样返回。
    """
    pos = text.find("</search>")
    if pos < 0:
        return text
    return text[: pos + len("</search>")]


class MultimodalSearchAgent:
    def __init__(
        self,
        model_path: str,
        retriever_url: str,
        topk: int = 3,
        max_new_tokens: int = 512,
        max_turns: int = 4,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
    ):
        print(f"[init] loading MLLM from {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype_map[torch_dtype],
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.retriever_url = retriever_url
        self.topk = topk
        self.max_new_tokens = max_new_tokens
        self.max_turns = max_turns

        tokenizer = self.processor.tokenizer
        # Qwen 系列的两个真正 stop token：</search>（用于触发检索）和 <|im_end|>（自然结束）
        self.stopping_criteria = StoppingCriteriaList([
            StopOnTokens(tokenizer, ["</search>", " </search>", "</search>\n", "<|im_end|>"])
        ])
        # 用于判断是否自然结束
        self.eos_token_ids = {
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.eos_token_id,
        }
        self.eos_token_ids.discard(None)

    def _build_initial_prompt(self, question: str, user_images: List[Image.Image]) -> str:
        """
        只在第一次调用 chat_template 渲染初始 prompt（含 user），
        后续多轮直接对返回的字符串做追加拼接。

        指令直接放在 user message 中（对齐原版 infer.py 风格），
        不使用 system message，避免模型忽略 system 指令。
        """
        instruction_text = USER_INSTRUCTION_TEMPLATE.format(
            topk=self.topk, question=question
        )
        user_content: List[dict] = [{"type": "image", "image": img} for img in user_images]
        user_content.append({"type": "text", "text": instruction_text})
        messages = [{"role": "user", "content": user_content}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_once(self, prompt_text: str, all_images: List[Image.Image]) -> Tuple[str, bool]:
        """
        对当前 prompt 字符串做一次生成，返回 (新生成的纯文本, 是否自然结束)。
        和 processor 一同传入累计图像列表，processor 会按 prompt 中 image_pad 的出现顺序匹配图。
        """
        inputs = self.processor(
            text=[prompt_text],
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        gen_text = self.processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        last_id = int(gen_tokens[-1].item()) if gen_tokens.numel() > 0 else None
        finished_naturally = last_id in self.eos_token_ids
        return gen_text, finished_naturally

    def run(self, question: str, user_image_paths: Optional[List[str]] = None) -> str:
        user_images: List[Image.Image] = []
        for path in user_image_paths or []:
            user_images.append(Image.open(path).convert("RGB"))

        # 初始 prompt：只渲染一次，后续追加纯文本（含图像占位符）
        prompt_text = self._build_initial_prompt(question, user_images)
        all_images: List[Image.Image] = list(user_images)
        full_assistant_text = ""

        print("\n================ START ================")
        print(f"[user question] {question}")
        if user_image_paths:
            print(f"[user images] {user_image_paths}")

        for turn in range(self.max_turns):
            gen_text, finished = self._generate_once(prompt_text, all_images)
            print(f"\n--- turn {turn} model output ---\n{gen_text}")
            full_assistant_text += gen_text

            # 自然结束 / 已经给出答案：把这段也拼进 prompt 后退出
            if finished or "<answer>" in gen_text:
                prompt_text += gen_text
                break

            # 没有 search 也没结束：模型不按格式，主动停
            query = extract_last_search_query(gen_text)
            if query is None:
                print("[info] no <search> in turn output, stop loop.")
                prompt_text += gen_text
                break

            # 截断到 </search>，把这段 assistant 输出拼回 prompt
            assistant_segment = truncate_after_search_tag(gen_text)
            prompt_text += assistant_segment

            print(f"\n[search] query = {query!r}")
            try:
                retrieved = call_retriever(self.retriever_url, query, self.topk)
            except Exception as e:
                print(f"[ERROR] retriever call failed: {e}")
                prompt_text += "\n<information>retriever unavailable</information>\n"
                continue

            information_text, info_images = build_information_text(retrieved)
            # 关键：information 段直接拼到 prompt 末尾，所有图像追加到 all_images
            # 让 processor 按 image_pad 出现顺序绑定图像
            prompt_text += information_text
            all_images.extend(info_images)
            print(f"[search] returned {len(info_images)} images")
        else:
            print("[info] reached max_turns without <answer>.")

        print("\n================ FULL ASSISTANT TRACE ================")
        print(full_assistant_text)
        return full_assistant_text


def parse_args():
    parser = argparse.ArgumentParser("Search-R1 Level 2 multimodal inference")
    parser.add_argument("--model_path", type=str,
                        default="/var/lib/container/dataset/yxqiu/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:18000/retrieve")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--image", type=str, action="append", default=None,
                        help="可选：用户输入图像路径，可重复传多次")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    agent = MultimodalSearchAgent(
        model_path=args.model_path,
        retriever_url=args.retriever_url,
        topk=args.topk,
        max_new_tokens=args.max_new_tokens,
        max_turns=args.max_turns,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    agent.run(args.question, args.image)


if __name__ == "__main__":
    main()
