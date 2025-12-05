#!/usr/bin/env python3
"""
LoRA / QLoRA finetuning script for Relation Radar (PR-0.3-04).

功能概述：
- 读取 `scripts/build_teacher_dataset.py` 生成的 JSONL：
    {"question": "...", "facts": "...", "ideal_answer": "...", "person_ids": [...]}
- 将 `facts + question` 拼成训练 prompt，以 `ideal_answer` 作为监督目标；
- 使用 LoRA/QLoRA 在本地 Qwen 基座模型上做参数高效微调；
- 输出一个 LoRA adapter 目录，可在推理时加载到基础模型上。

注意事项：
- 这是离线训练脚本，不会在 CI 中运行，只在你本地有 GPU 的环境里使用；
- 依赖项（需手动安装）：
    pip install "transformers>=4.42" "datasets" "accelerate" "peft" "bitsandbytes"
- QLoRA 建议使用 3B 级别的基础模型（例如：Qwen/Qwen2.5-3B-Instruct），
  显存 8–12GB 更为合适；也可以在 CPU 上训练但会非常慢。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from backend.llm.prompts import build_qa_rag_prompt


@dataclass
class TeacherExample:
    question: str
    facts: str
    answer: str


def load_teacher_dataset(path: Path) -> List[TeacherExample]:
    """Load teacher dataset from JSONL."""
    examples: List[TeacherExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data: Dict[str, object] = json.loads(line)
            q = str(data["question"]).strip()
            facts = str(data.get("facts") or "").strip()
            ans = str(data.get("ideal_answer") or "").strip()
            if not q or not ans:
                # 跳过不完整样本
                continue
            examples.append(TeacherExample(question=q, facts=facts, answer=ans))
    return examples


def build_text_pair(example: TeacherExample, *, add_eos: bool = True) -> str:
    """
    Build training text for a single example.

    我们重用 RAG 问答的 prompt 模板，让 LoRA 学习“在已有的 prompt 结构上如何输出更好的答案”。
    """
    prompt = build_qa_rag_prompt(question=example.question, context=example.facts)
    text = prompt.rstrip() + "\n\n理想回答：\n" + example.answer.strip()
    if add_eos:
        text += "</s>"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune a Qwen model with LoRA/QLoRA on teacher dataset.",
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
        "--output-dir",
        type=str,
        default="data/qwen2.5-3b-lora",
        help="Directory to save LoRA adapter.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, limit the number of training samples.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA adapter.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4bit QLoRA (requires bitsandbytes).",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_lora] loading teacher dataset from {dataset_path}")
    examples = load_teacher_dataset(dataset_path)
    if args.max_samples and args.max_samples > 0:
        examples = examples[: args.max_samples]
    print(f"[train_lora] using {len(examples)} examples")

    # 延迟导入重型依赖，避免没有安装时影响其它脚本使用。
    try:
        import datasets  # type: ignore
        import torch  # type: ignore
        from peft import (  # type: ignore
            LoraConfig,
            TaskType,
            get_peft_model,
        )
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - runtime environment issue
        raise SystemExit(
            "Missing training dependencies. Please install:\n"
            "  pip install 'transformers>=4.42' 'datasets' 'accelerate' 'peft' 'bitsandbytes'\n"
            f"Original error: {exc}",
        ) from exc

    # 构造 HF Dataset
    def gen() -> Dict[str, str]:
        for ex in examples:
            yield {"text": build_text_pair(ex)}

    hf_dataset = datasets.Dataset.from_generator(gen)

    print(f"[train_lora] loading base model {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if args.use_4bit else torch.float16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        enc = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=args.use_4bit,  # 在支持 bfloat16 的 GPU 上更友好
        fp16=not args.use_4bit,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    print("[train_lora] start training …")
    trainer.train()
    print("[train_lora] training finished, saving adapter …")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_lora] done. LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()

