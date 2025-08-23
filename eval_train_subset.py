import os
import sys
import json
import uuid
import random
import argparse
import tempfile
from pathlib import Path

import yaml


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from reinforced_cot.finetune.sft import SupervisedFineTuning
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on a subset of train set with a given LoRA ckpt.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to SFT yaml config (e.g., /.../configs/train/sft.yaml)")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to LoRA checkpoint dir (folder that contains adapter_config.json)")
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="Ratio of train set to sample for evaluation (0~1). Default: 0.2")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    return cfg["SFT"] if "SFT" in cfg else cfg


def is_lora_dir(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, "adapter_config.json"))


def jsonl_read(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def jsonl_write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_paths(cfg, cfg_dir: Path):
    
    def _abs(p):
        if p is None:
            return None
        p = str(p)
        return p if os.path.isabs(p) else str((cfg_dir / p).resolve())

    for key in ("model_path", "tokenizer_path", "image_dir", "test_image_dir", "log_dir"):
        if key in cfg:
            cfg[key] = _abs(cfg[key])

    pipe = cfg.setdefault("pipeline", {})
    for split in ("train", "val", "test"):
        sc = pipe.get(split)
        if not sc:
            continue
        key = f"{split}_file"
        if key in sc:
            sc[key] = _abs(sc[key])

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config_path)

    cfg_path = Path(args.config_path).resolve()
    cfg_dir = cfg_path.parent
    cfg = normalize_paths(cfg, cfg_dir)

    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("quantization", True)
    cfg.setdefault("log_dir", os.path.join(ROOT_DIR, "logs"))
    cfg.setdefault("max_input_length", 2048)
    cfg.setdefault("max_response_length", 1024)

    pipeline = cfg.setdefault("pipeline", {})
    train_cfg = pipeline.setdefault("train", {})
    test_cfg = pipeline.setdefault("test", {})

    train_file = train_cfg.get("train_file")
    if not train_file or not os.path.exists(train_file):
        raise FileNotFoundError(f"train_file not found in config: {train_file}")
    image_dir = cfg.get("image_dir")
    if not image_dir or not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found in config: {image_dir}")

    train_data = jsonl_read(train_file)
    n_total = len(train_data)
    n_sample = max(1, int(n_total * args.sample_ratio))

    random.seed(args.seed)
    subset = random.sample(train_data, n_sample)

    
    tmp_tag = uuid.uuid4().hex[:6]
    tmp_dir = os.path.join(tempfile.gettempdir(), f"eval_train_subset_{tmp_tag}")
    os.makedirs(tmp_dir, exist_ok=True)
    subset_path = os.path.join(tmp_dir, "train_subset.jsonl")
    jsonl_write(subset_path, subset)
    print(f"[INFO] 从训练集采样 {n_sample} 条到 {subset_path}")

    test_cfg["test_file"] = subset_path
    test_cfg["batch_size"] = args.batch_size
    test_cfg.setdefault("evaluating_epoch_freq", 1)
    cfg["test_image_dir"] = cfg.get("image_dir", cfg.get("SFT", {}).get("image_dir"))
    cfg["num_workers"] = 0
    if "SFT" in cfg:
        cfg["SFT"]["test_image_dir"] = cfg["test_image_dir"]
        cfg["SFT"]["num_workers"] = 0

    pipeline["val"] = {"batch_size": args.batch_size}

    trainer = SupervisedFineTuning(cfg)

    if args.ckpt_dir:
        if is_lora_dir(args.ckpt_dir):
            trainer.model = PeftModel.from_pretrained(trainer.model, args.ckpt_dir)
            trainer.model.eval()
            if trainer.accelerator.is_main_process:
                trainer.logger.INFO(f"Loaded LoRA adapter from {args.ckpt_dir}")
        else:
            if trainer.accelerator.is_main_process:
                trainer.logger.INFO(
                    f"[WARN] {args.ckpt_dir} does not look like a LoRA adapter folder (missing adapter_config.json). "
                    f"Skipping adapter loading."
                )

    acc = trainer.evaluate(tag="train_subset")
    if trainer.accelerator.is_main_process:
        trainer.logger.INFO(f"[Done] Train-subset accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
