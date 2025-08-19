import re


class AnswerProcessor:
    synonyms = {
        # 否定类
        "no": {
            "no", "not", "false", "nope", "negative",
            "cannot", "cant", "incorrect", "couldnt",
            "doesnt", "didnt", "isnt", "arent",
            "wasnt", "werent", "dont", "never", "none", "nothing"
        },
        # 肯定类
        "yes": {"yes", "true", "correct", "yep", "positive", "is", "are"},
        # 方向类
        "left": {"left", "west"},
        "right": {"right", "east"},
    }

    @classmethod
    def extract_answer_and_cot(cls,text: str):
        """
        Returns:
            extracted CoT and answer (can be a sentence rather than a word)
        """
        m_t = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.S|re.I)
        # 改进的 <answer> 提取逻辑
        m_a = re.search(r"<answer>(.*?)(?:</answer>|$)", text, flags=re.S|re.I)
        pred_answer = m_a.group(1).strip() if m_a else ""
        pred_think = m_t.group(1).strip() if m_t else ""
        return pred_think, pred_answer

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalize text by removing punctuation and converting to lowercase."""
        return re.sub(r'[^\w\s]', '', text.lower())

    @classmethod
    def is_match(cls, pred_answer: str, gt_answer: str) -> bool:

        pred_clean = cls.clean_text(pred_answer)
        gt_clean = cls.clean_text(gt_answer)

        # 直接匹配
        if gt_clean == pred_clean:
            return True

        # 优先检查同义词组
        for group in cls.synonyms.values():
            if gt_clean in group:
                if any(re.search(rf'\b{synonym}\b', pred_clean) for synonym in group):
                    return True
                else:
                    return False  # 属于同义词组但匹配失败时立即返回
    
        # 非同义词组的普通匹配
        return gt_clean in pred_clean.split()
    
