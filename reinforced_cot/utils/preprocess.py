import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import load_jsonl


class MultiModalQwenDataset(Dataset):
    def __init__(self, data_list, image_dir, processor, max_length=1024, split="train"):
        self.samples = data_list
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_dir, f"{item['imageId']}.jpg")
        image = Image.open(image_path).convert("RGB")

        instruction = item.get("instruction", item.get("question", ""))
        sys_prompt = DatasetPreprocessor.INSTRUCTION
        user_prompt = DatasetPreprocessor.QUESTION_TRIGGER + instruction

        if self.split in ("train", "val"):
            cot = item["cot"][0]["text"] if item.get("cot") and len(item["cot"]) > 0 else ""
            answer = item.get("answer", "")
            assistant_response = f"<think>\n{cot}\n</think>\n<answer>\n{answer}\n</answer>"
            full_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
            ]
            add_gen = False
        else:
            full_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
            ]
            add_gen = True

        text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=add_gen)
        model_inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        prompt_only_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor.tokenizer(text=[prompt_text], return_tensors="pt")
        prompt_length = prompt_inputs["input_ids"].shape[1]

        if self.split in ("train", "val"):
            labels = model_inputs["input_ids"].clone()
            labels[0, :prompt_length] = DatasetPreprocessor.LABEL_PAD_TOKEN_ID
        else:
            labels = torch.full_like(model_inputs["input_ids"], DatasetPreprocessor.LABEL_PAD_TOKEN_ID)

        out = {k: v.squeeze(0) for k, v in model_inputs.items()}
        out["labels"] = labels.squeeze(0)

        if self.split in ("val", "test"):
            out["image_id"] = item.get("imageId", None)
            out["images"] = image
            out["system_prompt"] = sys_prompt
            out["user_prompt"] = user_prompt
            out["answers"] = item.get("answer", "")

        return out


class DatasetPreprocessor:

    INSTRUCTION = (
        "You are an expert visual reasoning assistant. "
        "You will be provided with an image and some questions related to its content. "
        "First, generate a detailed, step-by-step chain of thought, "
        "and enclose this reasoning process within <think> tags. "
        "After your reasoning, provide a concise final answer and enclose it within <answer> tags.\n"
        "Your reasoning may involve these operators:\n"
        "- SELECT(object): Focus on a specific object in the image.\n"
        "- RELATE(object, relationship): Find objects related to the main object.\n"
        "- QUERY(attribute): Query specific attributes of the object.\n\n"
        "Strict output requirements:\n"
        "1. Do NOT use <tool_call> or any tags other than <think> and <answer>.\n"
        "2. Always wrap your reasoning inside <think>...</think>.\n"
        "3. Always wrap your final answer inside <answer>...</answer>.\n"
        "4. Present all reasoning steps and the final answer in English.\n"
    )

    QUESTION_TRIGGER = "\n### Question:\n"

    # padding constants
    LABEL_PAD_TOKEN_ID = -100

    @classmethod
    def prepare_datasets_and_data_loaders(cls, args, accelerator, processor, evaluate: bool = False):
        max_length = args.get("max_input_length", 1024)
        num_workers = args.get("num_workers", 0)

        def _prepare_split(split_name):
            if split_name not in args["pipeline"]:
                return None, None

            config = args["pipeline"][split_name]
            data_file_key = f"{split_name}_file"
            if data_file_key not in config:
                return None, None

            data = load_jsonl(config[data_file_key])
            for item in data:
                if "question" in item and "instruction" not in item:
                    item["instruction"] = item["question"]
            # img_dir = args["image_dir"]
            split_image_dir_key = f"{split_name}_image_dir"
            img_dir = args.get(split_image_dir_key, args.get("image_dir"))
            batch_size = config["batch_size"]

            dataset = MultiModalQwenDataset(
                data_list=data,
                image_dir=img_dir,
                processor=processor,
                max_length=max_length,
                split=split_name,
            )
            return dataset, batch_size

        with accelerator.main_process_first():
            train_dataset, batch_size_train = _prepare_split("train")
            val_dataset, batch_size_val = _prepare_split("val")
            test_dataset, batch_size_test = _prepare_split("test")

        # for training
        def collate_fn_train(batch):
            out = {}
            for k in batch[0].keys():
                if k in ("images", "user_prompt", "answers", "system_prompt"):
                    out[k] = [b[k] for b in batch]
                else:
                    if isinstance(batch[0][k], torch.Tensor):
                        out[k] = torch.stack([b[k] for b in batch])
                    else:
                        out[k] = [b[k] for b in batch]
            return out

        # for evaluation
        def collate_fn_eval(batch):
            out = {}
            for k in batch[0].keys():
                if k in ("images", "user_prompt", "instructions", "answers", "system_prompt"):
                    out[k] = [b[k] for b in batch]
                else:
                    v0 = batch[0][k]
                    if isinstance(v0, torch.Tensor):
                        try:
                            out[k] = torch.stack([b[k] for b in batch])
                        except RuntimeError:
                            out[k] = [b[k] for b in batch]
                    else:
                        out[k] = [b[k] for b in batch]
            return out

        collate_fn = collate_fn_eval if evaluate else collate_fn_train

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        return (
            (train_dataset, train_dataloader),
            (val_dataset, val_dataloader),
            (test_dataset, test_dataloader),
        )
