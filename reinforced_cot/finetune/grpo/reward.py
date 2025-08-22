import torch
from reinforced_cot.utils import AnswerProcessor


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Assigns a reward for adhering to the desired <think> and <answer> tag format.

    Args:
        completions (list): A batch of generated conversations from TRL.
        **kwargs: Catches other arguments from the trainer.

    Returns:
        A list of format compliance scores for each completion.
    """
    # Extract the final generated string from the conversation
    rewards = []

    for text in completions:
        score = 0.0
        # Assign 0.2 points for each of the four required tags
        if "<think>" in text:
            score += 0.2
        if "</think>" in text:
            score += 0.2
        if "<answer>" in text:
            score += 0.2
        if "</answer>" in text:
            score += 0.2
        rewards.append(score)

    return rewards


def correctness_and_consistency_reward(completions: list[str], ground_truth_answer: list[str], **kwargs) -> list[float]:
    """
    Assigns a combined reward for correctness and conditional consistency.

    - High reward if the answer is correct.
    - Bonus reward if the CoT is consistent WITH a correct answer.
    - Penalty if the answer is incorrect.
    - Penalty for non-concise answers.
    """
    rewards = []

    for text, gt_answer in zip(completions, ground_truth_answer):
        reward = 0.0
        pred_think, pred_answer = AnswerProcessor.extract_answer_and_cot(text)

        if not pred_answer:  # If no <answer> tag is found, there's no correctness to check.
            rewards.append(0.0)
            continue

        # 1. check for correctness
        is_correct = AnswerProcessor.is_match(pred_answer, gt_answer)

        if is_correct:
            reward = 2.0

            # 2. check for consistency if the answer is correct -> use gt to extract cot
            if pred_think:
                if AnswerProcessor.is_match(pred_think, gt_answer):
                    # bonus for a supporting Chain-of-Thought
                    reward += 0.5
                else:
                    # penalize reward hacking with inconsistent cot
                    reward -= 0.5

            # penalize long answers, even if correct
            if len(pred_answer.split()) > 3:
                reward -= 0.5
        else:
            # penalize for being explicitly wrong
            reward = -1.0

        rewards.append(reward)

    return rewards


def combined_reward(completions: list[str], **kwargs) -> torch.Tensor:

    # Calculate the score for each reward component
    format_scores = format_reward(completions=completions, **kwargs)
    main_scores = correctness_and_consistency_reward(completions=completions, **kwargs)

    # Combine the scores into a final reward
    # Max possible score: 0.8 (format) + 2.5 (correct + consistent) = 3.3
    # Min possible score (wrong answer): 0.8 (format) - 1.0 (correctness) = -0.2
    final_rewards = [f_score + m_score for f_score, m_score in zip(format_scores, main_scores)]

    return torch.tensor(final_rewards, dtype=torch.float32)
