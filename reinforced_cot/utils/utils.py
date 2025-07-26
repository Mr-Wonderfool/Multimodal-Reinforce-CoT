import json
import torch
import numpy as np
from typing import List, Dict, Optional


def load_jsonl(data_path, max_length: Optional[int] = None) -> List[Dict]:
    """
    Load data from jsonl files containing multiple lines of dictionaries.
    """
    assert data_path.endswith(".jsonl") or data_path.endswith(".json")
    problems = []
    with open(data_path, "r", encoding="utf-8") as file:
        if max_length is not None:
            for line in file.readlines()[:max_length]:
                problems.append(json.loads(line))
        else:
            for line in file.readlines():
                problems.append(json.loads(line))

    return problems


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
