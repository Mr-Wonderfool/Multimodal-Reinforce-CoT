import json
import torch
from collections import defaultdict
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from .utils import load_jsonl

# 读取jsonl文件
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# 每一条数据的读取与格式化，同时加载图片和文本内容
class MultiModalQwenDataset(Dataset):
    def __init__(self, data_list, image_dir, processor, max_length=1024):
        self.data = data_list # 样本列表
        self.image_dir = image_dir # 图片目录
        self.processor = processor # Qwen的AutoProcessor对象
        self.max_length = max_length # 最大长度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引读取一条数据
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['imageId']}.jpg")
        image = Image.open(image_path).convert("RGB")
        # 读取问题文本，cot，最终答案
        instruction = item["instruction"]
        cot = item["cot"][0]["text"] if item.get("cot") and len(item["cot"]) > 0 else ""
        answer = item.get("answer", "")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        # 组织target（推理链+答案），让模型学习如何从图片+问题输出推理链和答案
        target_text = f"{cot}\n\nFinal answer: {answer}"

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 使用processor处理文本和图片，生成模型输入
        model_inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        # 生成标签，labels是模型需要预测的目标文本
        labels = self.processor.tokenizer(
            target_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        out = {k: v.squeeze(0) for k, v in model_inputs.items()}
        out["labels"] = labels.squeeze(0)
        return out


# 多模态数据预处理器
class DatasetPreprocessor:

    instruction = (
    "You are a visual question answering expert. "
    "Your task is to analyze the given image carefully, understand the provided question and semantic chain, "
    "and generate a detailed step-by-step reasoning process (Chain of Thought) to answer the question based on the image content.\n\n"
    "Semantic Chain Operators Explanation:\n"
    "- SELECT(object): Focus on a specific object in the image\n"
    "- RELATE(object, relationship): Find objects related to the main object through a relationship\n"
    "- QUERY(attribute): Query specific attributes of the object\n"
    "\n### Question:\n"
    )
    cot_trigger = "\n\n### Answer reasoning:\n"
    answer_trigger = "\n\n### Final answer:\n"


    # padding constants
    LABEL_PAD_TOKEN_ID = -100

    def __init__(self):
        pass

    @classmethod
    def prepare_datasets_and_data_loaders(cls, args, accelerator, processor):
        """
        对应Qwen2.5-VL多模态场景，args需包含train/val/test的jsonl和图片目录。
        支持只传train/val或者train/test，也支持三者全传。
        """
        with accelerator.main_process_first():
            # 训练集
            train_data = load_jsonl(args["pipeline"]["train"]["train_file"])
            train_img_dir = args["pipeline"]["train"]["image_dir"]
            batch_size_train = args["pipeline"]["train"]["batch_size"]

            # 验证集
            val_data, val_img_dir, batch_size_val = None, None, None
            if "val" in args["pipeline"]:
                val_data = load_jsonl(args["pipeline"]["val"]["val_file"])
                val_img_dir = args["pipeline"]["val"]["image_dir"]
                batch_size_val = args["pipeline"]["val"]["batch_size"]

            # 测试集
            test_data, test_img_dir, batch_size_test = None, None, None
            if "test" in args["pipeline"]:
                test_data = load_jsonl(args["pipeline"]["test"]["test_file"])
                test_img_dir = args["pipeline"]["test"]["image_dir"]
                batch_size_test = args["pipeline"]["test"]["batch_size"]

            num_workers = args.get("num_workers", 2)
            max_length = args.get("max_input_length", 1024)

            train_dataset = MultiModalQwenDataset(
                data_list=train_data,
                image_dir=train_img_dir,
                processor=processor,
                max_length=max_length
            )

            if val_data is not None:
                val_dataset = MultiModalQwenDataset(
                    data_list=val_data,
                    image_dir=val_img_dir,
                    processor=processor,
                    max_length=max_length
                )
            else:
                val_dataset = None

            if test_data is not None:
                test_dataset = MultiModalQwenDataset(
                    data_list=test_data,
                    image_dir=test_img_dir,
                    processor=processor,
                    max_length=max_length
                )
            else:
                test_dataset = None

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
