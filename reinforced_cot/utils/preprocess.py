import re
import json
import torch
from collections import defaultdict
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from .utils import load_jsonl


class DatasetPreprocessor:

    instruction = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n### Question:\n"
    # problem-specific description
    starter_code = (
        lambda start_code: f"\n\n### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n{start_code}\n```"
    )
    cot_trigger = "\n\n### Answer reasoning:\n"
    answer_trigger = "\n\n### Therefore the answer code is:\n"

    # padding constants
    LABEL_PAD_TOKEN_ID = -100

    def __init__(self):
        pass

    @classmethod
    def prepare_datasets_and_data_loaders(cls, args, accelerator, tokenizer):
        """
        For returned data:
            `forward_kwargs` will be right-padded for supervised setting
            `generate_prefix_kwargs` will be left-padded for decoder-only models
                and should be used **for all generate method**
        """
        with accelerator.main_process_first():
            raw_dataset = DatasetDict(
                {
                    "train": Dataset.from_list(load_jsonl(args["pipeline"]["train"]["train_file"])),
                    "test": Dataset.from_list(load_jsonl(args["pipeline"]["test"]["test_file"])),
                }
            )

            # turn list[dict] into organized batch
            def tokenize_fn(batch, args, tokenizer):
                new_batch = defaultdict(list)
                all_keys = list(batch.keys())
                for item_values in zip(*(batch[k] for k in all_keys)):
                    item = {k: item_values[i] for i, k in enumerate(all_keys)}

                    question_id = item["question_id"]
                    problem_description = item["problem_description"]
                    answer_cot = item["chain_of_thought"]
                    start_code = item["starter_code"]
                    # ! only use the code for response, ignore natural language explanations
                    response_code = item["response_code"]
                    # for compiler input
                    prompt = item["prompt"]
                    test_func = item["test"]

                    input = f"{cls.instruction}{problem_description}{cls.starter_code(start_code)}{cls.cot_trigger}"
                    # code + explanations as output
                    output = f"{answer_cot}{cls.answer_trigger}{response_code}"
                    prefix_text = (
                        f"{cls.instruction}{problem_description}{cls.starter_code(start_code)}{cls.cot_trigger}"
                    )

                    # encode tokens to input_ids
                    input_encode = tokenizer(input, add_special_tokens=False)
                    output_encode = tokenizer(output, add_special_tokens=False)
                    prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                    input_ids = input_encode["input_ids"] + output_encode["input_ids"] + [tokenizer.eos_token_id]
                    labels = (
                        [cls.LABEL_PAD_TOKEN_ID] * len(input_encode["input_ids"])
                        + output_encode["input_ids"]
                        + [tokenizer.eos_token_id]
                    )
                    # all valid entries at the moment
                    attention_mask = [1] * len(input_ids)
                    prefix = prefix_encode["input_ids"]
                    prefix_attention_mask = prefix_encode["attention_mask"]

                    # Truncation according to max length in arguments
                    input_ids_max_length = len(input_ids)
                    args_max_length = args["max_input_length"]
                    input_ids = input_ids[:args_max_length]
                    labels = labels[:args_max_length]
                    attention_mask = attention_mask[:args_max_length]
                    prefix = prefix[:args_max_length]
                    prefix_attention_mask = prefix_attention_mask[:args_max_length]

                    new_batch["input_ids"].append(input_ids)
                    new_batch["labels"].append(labels)
                    new_batch["attention_mask"].append(attention_mask)
                    new_batch["prefix"].append(prefix)
                    new_batch["prefix_attention_mask"].append(prefix_attention_mask)

                    new_batch["question_id"].append(question_id)
                    new_batch["answer_cot"].append(answer_cot)
                    new_batch["input_ids_max_length"].append(input_ids_max_length)
                    
                    # for compiler input
                    new_batch["prompt"].append(prompt)
                    new_batch["test_func"].append(test_func)

                return new_batch

            tokenized_dataset = DatasetDict(
                {
                    mode: dataset.map(
                        lambda batch: tokenize_fn(batch, args, tokenizer),
                        batched=True,
                        remove_columns=dataset.column_names,
                    )
                    for mode, dataset in raw_dataset.items()
                }
            )

        def collate_fn(batch, tokenizer):
            max_input_length = max([len(item["input_ids"]) for item in batch])
            max_target_length = max([len(item["labels"]) for item in batch])
            max_prefix_length = max([len(item["prefix"]) for item in batch])

            input_ids = []
            attention_mask = []
            labels, labels_left_padded = [], []
            prefix_left_padded = []
            prefix_attention_mask_left_padded = []
            
            prompts = []
            test_funcs = []
            question_ids = []

            for item in batch:
                input_ids.append(
                    item["input_ids"] + [tokenizer.pad_token_id] * (max_input_length - len(item["input_ids"]))
                )
                attention_mask.append(item["attention_mask"] + [0] * (max_input_length - len(item["attention_mask"])))
                labels.append(item["labels"] + [cls.LABEL_PAD_TOKEN_ID] * (max_target_length - len(item["labels"])))

                labels_left_padded.append(
                    [cls.LABEL_PAD_TOKEN_ID] * (max_target_length - len(item["labels"])) + item["labels"]
                )
                prefix_left_padded.append(
                    [tokenizer.pad_token_id] * (max_prefix_length - len(item["prefix"])) + item["prefix"]
                )
                prefix_attention_mask_left_padded.append(
                    [0] * (max_prefix_length - len(item["prefix_attention_mask"])) + item["prefix_attention_mask"]
                )
                
                prompts.append(item["prompt"])
                test_funcs.append(item["test_func"])
                question_ids.append(item["question_id"])

            # for model training (right padded for supervised training)
            # add question id for test time logging
            forward_kwargs = {
                "question_ids": question_ids,
                "input_ids": torch.LongTensor(input_ids),
                "attention_mask": torch.BoolTensor(attention_mask),
                "labels": torch.LongTensor(labels),
            }

            # for inferene (left padded)
            generate_prefix_kwargs = {
                "question_ids": question_ids,
                "input_ids": torch.LongTensor(prefix_left_padded),
                "attention_mask": torch.BoolTensor(prefix_attention_mask_left_padded),
                "labels": torch.LongTensor(labels_left_padded),
                # pass import statements and test functions for compiler feedback
                "prompts": prompts,
                "test_funcs": test_funcs,
            }

            return {
                "forward_kwargs": forward_kwargs,
                "generate_prefix_kwargs": generate_prefix_kwargs,
            }

        train_dataloader = DataLoader(
            tokenized_dataset["train"],
            batch_size=args["pipeline"]["train"]["batch_size"],
            num_workers=args["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )
        test_dataloader = DataLoader(
            tokenized_dataset["test"],
            batch_size=args["pipeline"]["test"]["batch_size"],
            num_workers=args["num_workers"],
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )

        return (tokenized_dataset["train"], train_dataloader), (tokenized_dataset["test"], test_dataloader)


class CodePreprocessor:

    @staticmethod
    def remove_python_comments(code: str) -> str:
        """
        Remove **single line** comment (marked with #) from given code
        """
        uncommented_code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        # Remove excessive blank lines that result from comment removal
        return re.sub(r"\n\s*\n", "\n", uncommented_code).strip()

    @staticmethod
    def extract_code_from_response(
        full_response: str,
        remove_comments: bool = True,
        wrap_extracted_code: bool = False,
    ):
        """
        Parameter:
            full_response: str, the response text containing code blocks
            remove_comments: bool, whether to remove comments from the extracted code
            wrap_extracted_code: bool, whether to wrap the extracted code in code block delimiters
        """
        # match code pattern from response
        code_pattern = re.compile(r"```python\n([\s\S]*?)```")
        match = code_pattern.search(full_response)
        if match:
            code_content = match.group(1)
            if remove_comments:
                code_content = CodePreprocessor.remove_python_comments(code_content)
            # re-wrap the code
            if wrap_extracted_code:
                extracted_code = f"```python\n{code_content}```"
            else:
                extracted_code = code_content
            return extracted_code

        else:
            return ""

    @staticmethod
    def extract_code_from_jsonl(
        input_file_path: str,
        output_file_path: str,
        response_field: str = "response",
        code_field: str = "response_code",
        remove_comments: bool = True,
    ):
        # match code pattern from response
        code_pattern = re.compile(r"```python\n([\s\S]*?)```")
        input_data = load_jsonl(input_file_path)
        # record number for processed lines and expected lines
        total_lines = len(input_data)
        processed_lines = 0
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for each_dict in input_data:
                match = code_pattern.search(each_dict[response_field])
                if match:
                    code_content = match.group(1)
                    if remove_comments:
                        code_content = CodePreprocessor.remove_python_comments(code_content)
                    # re-wrap the code
                    extracted_code = f"```python\n{code_content}```"

                    each_dict[code_field] = extracted_code

                    outfile.write(json.dumps(each_dict) + "\n")

                    processed_lines += 1

                else:
                    print(
                        f"Code not found in item {each_dict['question_id']} from {input_file_path}, response length: {len(each_dict[response_field])}"
                    )
                    each_dict[code_field] = ""

        print(f"Processing finished, total lines: {total_lines}, processed lines: {processed_lines}")
