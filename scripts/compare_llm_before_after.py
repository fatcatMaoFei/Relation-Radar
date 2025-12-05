#!/usr/bin/env python3
"""
Compare base vs LoRA-finetuned model on teacher examples (PR-0.3-04).

用途：
- 给定一份 teacher 数据集（或单独的测试集），以及：
    - 一个基础模型（未微调版本）；
    - 一个带 LoRA adapter 的模型目录；
- 对同一批问题打印两份回答，方便人工主观评估：
    - base_answer
    - lora_answer

和训练脚本一样，这里依赖 Transformers / peft 等，需要你在本地主动安装：
    pip install "transformers>=4.42" "accelerate" "peft" "bitsandbytes"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from backend.llm.prompts import build_qa_rag_prompt


def load_examples(path: Path, max_examples: int) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            q = str(data["question"]).strip()
            facts = str(data.get("facts") or "").strip()
            ans = str(data.get("ideal_answer") or "").strip()
            if not q:
                continue
            items.append({"question": q, "facts": facts, "answer": ans})
            if max_examples and len(items) >= max_examples:
                break
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base vs LoRA-finetuned model on teacher dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/teacher_dataset.jsonl",
        help="Path to teacher dataset JSONL.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base HF model name or local path.",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (output of train_lora_qwen.py).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to compare.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    examples = load_examples(dataset_path, args.num_examples)
    print(f"[compare] loaded {len(examples)} examples from {dataset_path}")

    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies. Please install:\n"
            "  pip install 'transformers>=4.42' 'accelerate' 'peft' 'bitsandbytes'\n"
            f"Original error: {exc}",
        ) from exc

    print(f"[compare] loading base model {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"[compare] loading LoRA adapter from {args.lora_adapter}")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_adapter)

    def generate(model, prompt: str) -> str:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单切掉 prompt 部分，只保留模型新增的内容
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()

    for idx, ex in enumerate(examples, 1):
        prompt = build_qa_rag_prompt(question=ex["question"], context=ex["facts"])
        print("=" * 80)
        print(f"[Example {idx}] Question: {ex['question']}")
        print("-" * 80)
        print("FACTS:")
        print(ex["facts"])
        print("-" * 80)
        print("TEACHER (ideal_answer):")
        print(ex["answer"])
        print("-" * 80)

        print("[Base model answer]:")
        base_answer = generate(base_model, prompt)
        print(base_answer)

        print("\n[LoRA model answer]:")
        lora_answer = generate(lora_model, prompt)
        print(lora_answer)
        print("=" * 80)


if __name__ == "__main__":
    main()

