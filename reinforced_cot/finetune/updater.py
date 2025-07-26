import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any


class PPOUpdater:
    def __init__(
        self,
        model,  # 策略和价值模型
        optimizer: AdamW,
        accelerator,
        clip_grad_norm,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ):
        """
        初始化 PPO 更新器

        参数:
            model: 包含策略头和价值头的模型。
                   - model.forward(...) 应返回 (logits, values)
            optimizer: 优化器，例如 AdamW。
            ppo_epochs: 在同一批数据上更新模型的次数。
            clip_epsilon: PPO 裁剪范围的超参数 epsilon。
            gamma: 折扣因子。
            lambda_gae: GAE 的 lambda 参数。
            vf_coef: 价值损失的系数。
            ent_coef: 熵奖励的系数。
        """
        self.model = model
        self.optimizer = optimizer
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.accelerator = accelerator
        self.clip_grad_norm = clip_grad_norm

    def update(self, rollouts: Dict[str, Any]):
        query_tensors = rollouts["query_tensors"]
        response_tensors = rollouts["response_tensors"]
        old_log_probs = rollouts["log_probs"] # 形状正确 (B, L)
        old_values = rollouts["values"]       # 形状正确 (B, L)
        rewards = rollouts["rewards"]         # 形状正确 (B, L)
        attention_mask = rollouts["attention_mask"] # 形状正确 (B, L)

        advantages, returns = self._compute_gae_and_returns(old_values, rewards, attention_mask)

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()

            input_ids = torch.cat([query_tensors, response_tensors], dim=1)
            mask = torch.cat([rollouts["query_mask"], attention_mask], dim=1)

            logits, current_full_values = self.model(input_ids, attention_mask=mask)

            response_logits = logits[:, query_tensors.shape[1] - 1 : -1, :]
            new_log_probs = F.log_softmax(response_logits, dim=-1)
            new_log_probs = torch.gather(new_log_probs, 2, response_tensors.unsqueeze(-1)).squeeze(-1)
            new_log_probs = new_log_probs * attention_mask # 形状保持为 (B, L)

            entropy = (torch.distributions.Categorical(logits=response_logits).entropy() * attention_mask).sum(dim=1)


            current_values = current_full_values[:, query_tensors.shape[1] - 1 : -1]

            value_loss = F.mse_loss(current_values * attention_mask, returns * attention_mask)

            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * attention_mask).sum() / attention_mask.sum()

            total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

    def _compute_gae_and_returns(self, values: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor):
        """
        使用 GAE (Generalized Advantage Estimation) 计算优势和回报。
        从序列的末尾向前计算。
        """
        # values 的维度是 (batch_size, seq_len)
        # rewards 的维度是 (batch_size, seq_len)，通常只有最后一个时间步有非零值

        last_gae_lam = 0
        advantages_reversed = []

        # 扩展 values 以包含最终状态的价值（这里简化为0）
        values_extended = torch.cat([values, torch.zeros_like(values[:, :1])], dim=1)

        # 从后往前遍历序列
        for t in reversed(range(rewards.shape[1])):
            # 获取 V(s_t+1) 和 V(s_t)
            v_t_plus_1 = values_extended[:, t + 1]
            v_t = values_extended[:, t]

            # 计算 TD 误差 delta
            delta = rewards[:, t] + self.gamma * v_t_plus_1 - v_t

            # 计算 GAE
            last_gae_lam = delta + self.gamma * self.lambda_gae * last_gae_lam
            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 使用掩码确保 padding token 的优势为0
        advantages = advantages * mask

        # 计算回报 (Returns = Advantages + Values)
        returns = advantages + values

        # 标准化优势函数，可以稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns
