"""
PPO (Proximal Policy Optimization) training script for RL-based token pruning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model.base_mllm import get_mllm
from pruner.rl_pruner import RLPruner
from data.base_loader import get_data_loader


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories.
    """
    def __init__(self):
        self.states_visual = []
        self.states_query = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []  # Valid token masks

    def add(self, state_visual, state_query, action, log_prob, reward, value, done, mask):
        self.states_visual.append(state_visual)
        self.states_query.append(state_query)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self):
        self.states_visual.clear()
        self.states_query.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.masks.clear()

    def get(self):
        return (
            self.states_visual,
            self.states_query,
            self.actions,
            self.log_probs,
            self.rewards,
            self.values,
            self.dones,
            self.masks
        )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: list of rewards
        values: list of value estimates
        dones: list of done flags
        gamma: discount factor
        lam: GAE lambda
    Returns:
        advantages: list of advantages
        returns: list of returns
    """
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    return advantages, returns


def compute_task_reward(mllm, visual_features, query_components, ground_truth, config):
    """
    Compute task-specific reward by evaluating model performance.

    Args:
        mllm: vision-language model
        visual_features: pruned visual features [1, K, D]
        query_components: dict with text embeddings
        ground_truth: ground truth answer/bbox
        config: config object
    Returns:
        reward: scalar reward
    """
    try:
        # Combine visual features with text embeddings
        combined_embeddings = torch.cat([
            query_components['text_embeds_part1'],
            visual_features,
            query_components['text_embeds_part2']
        ], dim=1)

        attention_mask = torch.ones(
            (1, combined_embeddings.shape[1]),
            dtype=torch.long,
            device=mllm.device
        )

        # Generate answer
        generated_answer = mllm.generate_answer(combined_embeddings, attention_mask, max_new_tokens=20)

        # Compute reward based on task
        if 'bbox' in ground_truth and ground_truth['bbox'] is not None:
            # For bbox tasks, use IoU
            from evaluator.evaluator import _parse_bbox_from_text, _compute_iou
            pred_bbox = _parse_bbox_from_text(generated_answer)
            if pred_bbox is None:
                reward = 0.0
            else:
                iou = _compute_iou(pred_bbox, tuple(ground_truth['bbox']))
                reward = iou
        else:
            # For text tasks, use exact match
            gt_answer = ground_truth.get('answer', '')
            if gt_answer.lower() in generated_answer.lower():
                reward = 1.0
            else:
                reward = 0.0

        return reward

    except Exception as e:
        print(f"Error computing task reward: {e}")
        return 0.0


def collect_rollouts(agent, mllm, data_loader, num_rollouts, max_steps, config, device):
    """
    Collect rollout trajectories using current policy.

    Args:
        agent: RL agent
        mllm: vision-language model
        data_loader: data loader
        num_rollouts: number of rollouts to collect
        max_steps: maximum pruning steps per rollout
        config: config object
        device: device
    Returns:
        buffer: RolloutBuffer with collected trajectories
    """
    buffer = RolloutBuffer()
    agent.eval()

    samples = data_loader.get_train_samples()
    alpha = getattr(config, 'PPO_REWARD_ALPHA', 1.0)
    beta = getattr(config, 'PPO_REWARD_BETA', 0.1)
    step_discount = getattr(config, 'RL_STEP_DISCOUNT', 0.5)

    print(f"Collecting {num_rollouts} rollouts...")

    for rollout_idx in tqdm(range(num_rollouts)):
        sample_idx = rollout_idx % len(samples)
        sample = samples[sample_idx]

        try:
            # Get components from MLLM
            components = mllm.get_components_for_env(sample['image'], sample['question'])
            if components is None:
                continue

            visual_features = components['original_visual_features']  # [1, N, D]
            query_embedding = components['query_embeddings']  # [1, 1, D]

            # Multi-step pruning trajectory
            current_features = visual_features.clone()
            prev_num_tokens = current_features.shape[1]
            prev_task_score = 0.0

            for step in range(max_steps):
                # Get policy output
                with torch.no_grad():
                    probs, value = agent(current_features, query_embedding, return_value=True)

                # Sample action
                discount_factor = step_discount ** step
                actions, log_probs = agent.sample_action(probs, step_discount=discount_factor)

                # Apply action to prune tokens
                mask = actions[0]  # [K]
                kept_indices = torch.where(mask > 0)[0]

                if len(kept_indices) == 0:
                    # Keep at least one token
                    kept_indices = torch.tensor([probs[0].argmax()], device=device)

                next_features = current_features[:, kept_indices, :]
                num_kept = next_features.shape[1]

                # Compute reward
                # Task reward: change in task performance
                task_score = compute_task_reward(
                    mllm, next_features, components,
                    {'bbox': sample.get('bbox'), 'answer': sample.get('answer')},
                    config
                )
                task_reward = alpha * (task_score - prev_task_score)

                # Efficiency reward: compression ratio
                compression_ratio = 1.0 - (num_kept / prev_num_tokens)
                eff_reward = beta * compression_ratio

                reward = task_reward + eff_reward

                # Store transition
                done = (step == max_steps - 1) or (num_kept <= 1)

                buffer.add(
                    state_visual=current_features[0].cpu(),
                    state_query=query_embedding[0].cpu(),
                    action=actions[0].cpu(),
                    log_prob=log_probs[0].cpu(),
                    reward=reward,
                    value=value[0].item(),
                    done=float(done),
                    mask=torch.ones(current_features.shape[1]).cpu()
                )

                # Update for next step
                current_features = next_features
                prev_num_tokens = num_kept
                prev_task_score = task_score

                if done:
                    break

        except Exception as e:
            print(f"Error in rollout {rollout_idx}: {e}")
            continue

    return buffer


def ppo_update(agent, buffer, optimizer_actor, optimizer_critic, config, device):
    """
    Perform PPO update.

    Args:
        agent: RL agent
        buffer: RolloutBuffer
        optimizer_actor: optimizer for policy
        optimizer_critic: optimizer for value network
        config: config object
        device: device
    Returns:
        metrics: dict of training metrics
    """
    # Get data from buffer
    states_visual, states_query, actions, old_log_probs, rewards, values, dones, masks = buffer.get()

    # Compute advantages and returns
    advantages, returns = compute_gae(
        rewards, values, dones,
        gamma=getattr(config, 'PPO_GAMMA', 0.99),
        lam=getattr(config, 'PPO_LAM', 0.95)
    )

    # Convert to tensors
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO hyperparameters
    clip_epsilon = getattr(config, 'PPO_CLIP_EPSILON', 0.2)
    c1 = getattr(config, 'PPO_VALUE_COEF', 0.5)
    c2 = getattr(config, 'PPO_ENTROPY_COEF', 0.01)
    ppo_epochs = getattr(config, 'PPO_EPOCHS', 4)
    mini_batch_size = getattr(config, 'PPO_MINI_BATCH_SIZE', 16)

    # Training metrics
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    agent.train()

    for epoch in range(ppo_epochs):
        # Create mini-batches
        num_samples = len(states_visual)
        indices = np.random.permutation(num_samples)

        for start in range(0, num_samples, mini_batch_size):
            end = min(start + mini_batch_size, num_samples)
            batch_indices = indices[start:end]

            # Prepare batch
            batch_visual = torch.stack([states_visual[i] for i in batch_indices]).to(device)
            batch_query = torch.stack([states_query[i] for i in batch_indices]).to(device)
            batch_actions = torch.stack([actions[i] for i in batch_indices]).to(device)
            batch_old_log_probs = torch.stack([old_log_probs[i] for i in batch_indices]).to(device)
            batch_advantages = advantages[batch_indices].to(device)
            batch_returns = returns[batch_indices].to(device)

            # Forward pass
            probs, values_pred = agent(batch_visual, batch_query, return_value=True)

            # Compute new log probs
            new_log_probs = agent.get_action_log_probs(probs, batch_actions)

            # Compute ratio
            ratio = torch.exp(new_log_probs.sum(dim=1) - batch_old_log_probs.sum(dim=1))

            # Compute surrogate losses
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values_pred, batch_returns)

            # Entropy bonus
            entropy = agent.get_entropy(probs)

            # Total loss
            loss = policy_loss + c1 * value_loss - c2 * entropy

            # Update actor
            optimizer_actor.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
            optimizer_actor.step()

            # Update critic
            optimizer_critic.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.value.parameters(), max_norm=0.5)
            optimizer_critic.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    metrics = {
        'policy_loss': total_policy_loss / num_updates,
        'value_loss': total_value_loss / num_updates,
        'entropy': total_entropy / num_updates,
        'avg_reward': np.mean(rewards),
        'avg_return': np.mean(returns.numpy())
    }

    return metrics


def main():
    """
    Main function for PPO training.
    """
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    logger.info("=== PPO Training ===")

    # Load MLLM
    logger.info("Loading MLLM...")
    mllm = get_mllm(config)

    # Load data
    logger.info("Loading data...")
    data_loader = get_data_loader(config)

    # Create RL agent
    logger.info("Creating RL agent...")
    rl_pruner = RLPruner(mllm, config)
    agent = rl_pruner.model

    # Load LfD checkpoint if available
    lfd_checkpoint_path = getattr(config, 'LFD_CHECKPOINT_PATH', None)
    if lfd_checkpoint_path and os.path.exists(lfd_checkpoint_path):
        logger.info(f"Loading LfD checkpoint from {lfd_checkpoint_path}")
        checkpoint = torch.load(lfd_checkpoint_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("No LfD checkpoint found. Starting from scratch.")

    # Setup optimizers (separate for actor and critic)
    lr_actor = getattr(config, 'PPO_LR_ACTOR', 1e-5)
    lr_critic = getattr(config, 'PPO_LR_CRITIC', 3e-5)

    optimizer_actor = optim.AdamW(
        list(agent.shared_attention.parameters()) + list(agent.policy.parameters()),
        lr=lr_actor
    )
    optimizer_critic = optim.AdamW(
        list(agent.shared_attention.parameters()) + list(agent.value.parameters()),
        lr=lr_critic
    )

    # Training loop
    num_epochs = getattr(config, 'PPO_NUM_EPOCHS', 200)
    rollout_batch_size = getattr(config, 'PPO_ROLLOUT_BATCH_SIZE', 64)
    max_steps = getattr(config, 'PPO_MAX_STEPS', 3)

    logger.info(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Collect rollouts
        buffer = collect_rollouts(
            agent, mllm, data_loader,
            num_rollouts=rollout_batch_size,
            max_steps=max_steps,
            config=config,
            device=config.DEVICE
        )

        # PPO update
        metrics = ppo_update(
            agent, buffer, optimizer_actor, optimizer_critic,
            config, config.DEVICE
        )

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Policy Loss: {metrics['policy_loss']:.4f}, "
            f"Value Loss: {metrics['value_loss']:.4f}, "
            f"Entropy: {metrics['entropy']:.4f}, "
            f"Avg Reward: {metrics['avg_reward']:.4f}, "
            f"Avg Return: {metrics['avg_return']:.4f}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.LOG_DIR, f'ppo_checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("PPO training completed!")


if __name__ == "__main__":
    main()
