#!/usr/bin/env python
"""LoRA fine-tune for structured extraction.

    python src/train_lora.py                      # defaults, measured locally
    python src/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3

Device: CUDA if present, else Apple MPS, else CPU. The measured run in the
README is MPS on an M4; the notebook defaults to CUDA on a Colab T4.

**Loss is computed on the completion only.** The prompt tokens are masked to
-100 so the model is not also trained to reproduce the instruction and the
ticket. Training on the full sequence is the most common mistake in a
fine-tuning script and it quietly wastes most of the gradient signal on text
the model will always be *given*, never asked to generate.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import build_messages, target_json  # noqa: E402

IGNORE = -100


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Hyperparams:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # All four attention projections plus the three MLP projections. Attention-only
    # (q,v) is the cheaper classic choice; including the MLP is what current
    # practice recommends when the task changes the *output format*, because the
    # MLP is where token-level formatting behaviour lives.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 2
    max_length: int = 512
    warmup_ratio: float = 0.03  # converted to steps; transformers 5 dropped warmup_ratio
    seed: int = 20261119


class ExtractionDataset(Dataset):
    """Tokenise once, mask the prompt, keep the completion."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.features: list[dict] = []
        truncated = 0

        for row in rows:
            messages = build_messages(row["ticket"])
            # transformers 5.x returns a BatchEncoding here, not a plain list.
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            prompt_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            completion_ids = tokenizer(
                target_json(row["label"]) + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = prompt_ids + completion_ids
            if len(input_ids) > max_length:
                truncated += 1
                continue  # drop rather than truncate: a cut JSON target teaches malformed output
            labels = [IGNORE] * len(prompt_ids) + completion_ids

            self.features.append({"input_ids": input_ids, "labels": labels})

        if truncated:
            print(f"  dropped {truncated} examples longer than {max_length} tokens")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


def collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad, and mask the padding out of the loss as well as attention."""
    longest = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []

    for item in batch:
        pad = longest - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        labels.append(item["labels"] + [IGNORE] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def peak_memory_gb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1024**3
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    defaults = Hyperparams()
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--rank", type=int, default=defaults.rank)
    parser.add_argument("--alpha", type=int, default=defaults.alpha)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--out", type=Path, default=ROOT / "adapters" / "qwen05b-triage-lora")
    args = parser.parse_args()

    hp = Hyperparams(
        model=args.model,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    device = pick_device()
    # bf16 on CUDA and MPS; fp32 on CPU where bf16 is slower than it is smaller.
    dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    torch.manual_seed(hp.seed)

    print(f"device={device} dtype={dtype} model={hp.model}")

    tokenizer = AutoTokenizer.from_pretrained(hp.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(hp.model, dtype=dtype)
    model.config.use_cache = False  # incompatible with gradient checkpointing / training
    model.to(device)

    lora = LoraConfig(
        r=hp.rank,
        lora_alpha=hp.alpha,
        lora_dropout=hp.dropout,
        target_modules=list(hp.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    train_rows = read_jsonl(ROOT / "data" / "train.jsonl")
    dataset = ExtractionDataset(train_rows, tokenizer, hp.max_length)
    print(f"train examples: {len(dataset)}")

    # transformers 5 removed `warmup_ratio`, so compute the step count here.
    steps_per_epoch = max(1, len(dataset) // (hp.batch_size * hp.grad_accum))
    warmup_steps = max(1, int(steps_per_epoch * hp.epochs * hp.warmup_ratio))

    args_tr = TrainingArguments(
        output_dir=str(ROOT / "runs" / "train"),
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.grad_accum,
        learning_rate=hp.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=hp.seed,
        bf16=(device == "cuda"),
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    card = {
        "hyperparams": asdict(hp),
        "device": device,
        "dtype": str(dtype),
        "platform": f"{platform.platform()} ({platform.machine()})",
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "train_examples": len(dataset),
        "warmup_steps": warmup_steps,
        "train_seconds": round(elapsed, 1),
        "final_train_loss": round(result.training_loss, 4),
        "peak_memory_gb": (
            round(peak_memory_gb(device), 2) if peak_memory_gb(device) is not None else None
        ),
        "adapter_bytes": sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()),
    }
    (args.out / "training_card.json").write_text(json.dumps(card, indent=2))

    print(f"\nsaved adapter -> {args.out.relative_to(ROOT)}")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
