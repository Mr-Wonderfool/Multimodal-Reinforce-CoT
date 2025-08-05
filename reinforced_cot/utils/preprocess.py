import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .utils import load_jsonl


# 每一条数据的读取与格式化，同时加载图片和文本内容
class MultiModalQwenDataset(Dataset):
    def __init__(self, data_list, image_dir, processor, max_length=1024):
        self.samples = data_list  # 样本列表
        self.image_dir = image_dir  # 图片目录
        self.processor = processor  # Qwen的AutoProcessor对象
        self.max_length = max_length  # 最大长度
        self.instruction = DatasetPreprocessor.INSTRUCTION
        self.label_pad_token_id = DatasetPreprocessor.LABEL_PAD_TOKEN_ID

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 根据索引读取一条数据
        item = self.samples[idx]
        image_path = os.path.join(self.image_dir, f"{item['imageId']}.jpg")
        image = Image.open(image_path).convert("RGB")

        # 读取问题文本，cot，最终答案
        user_prompt = self.instruction + item["instruction"]
        cot = item["cot"][0]["text"] if item.get("cot") and len(item["cot"]) > 0 else ""
        answer = item.get("answer", "")
        assistant_response = f"<think>{cot}</think>\n<answer>{answer}</answer>"

        user_content = [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]
        prompt_only_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
        ]

        text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        model_inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        prompt_text = self.processor.apply_chat_template(
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor.tokenizer(text=[prompt_text], return_tensors="pt")
        prompt_length = prompt_inputs["input_ids"].shape[1]

        labels = model_inputs["input_ids"].clone()
        # mask out the part corresponding to prompts
        labels[0, :prompt_length] = self.label_pad_token_id
        out = {k: v.squeeze(0) for k, v in model_inputs.items()}
        out["labels"] = labels.squeeze(0)
        return out


# 多模态数据预处理器
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
        "\n### Question:\n"
    )
    # padding constants
    LABEL_PAD_TOKEN_ID = -100

    @classmethod
    def prepare_datasets_and_data_loaders(cls, args, accelerator, processor):
        """
        对应Qwen2.5-VL多模态场景，args需包含train/val/test的jsonl和图片目录。
        支持只传train/val或者train/test，也支持三者全传。
        """
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
            img_dir = args["image_dir"]
            batch_size = config["batch_size"]

            dataset = MultiModalQwenDataset(
                data_list=data,
                image_dir=img_dir,
                processor=processor,
                max_length=max_length,
            )
            return dataset, batch_size

        with accelerator.main_process_first():
            train_dataset, batch_size_train = _prepare_split("train")
            val_dataset, batch_size_val = _prepare_split("val")
            test_dataset, batch_size_test = _prepare_split("test")

        def collate_fn(batch):
            out = {}
            for k in batch[0].keys():
                out[k] = torch.stack([b[k] for b in batch])
            return out

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

        # 返回全部数据集与加载器
        return (
            (train_dataset, train_dataloader),
            (val_dataset, val_dataloader),
            (test_dataset, test_dataloader),
        )
