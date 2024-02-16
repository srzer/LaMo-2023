import math
import utils
import torch
import gym
import d4rl_atari
import numpy as np
from dotmap import DotMap
from omegaconf import OmegaConf
from model import DecisionTransformer
from hydra.utils import instantiate
from copy import deepcopy
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from gym.vector import SyncVectorEnv
from utils import TorchDeque
import wandb
from tqdm import tqdm
from get_nlp_datasets import get_dataset
from itertools import cycle
from get_hns import get_normalized_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLRScheduler(_LRScheduler): #follows the setup of original DT atari code
    def __init__(self, optimizer, warmup_tokens, final_tokens, last_epoch=-1):
        self.tokens = 0
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
        else:
            # cosine learning rate decay
            progress = float(self.tokens - self.warmup_tokens) / float(max(1, self.final_tokens - self.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [lr * lr_mult for lr in self.base_lrs]

    def step(self, tokens=None):
        if tokens is not None:
            self.tokens += tokens
        super().step()

@torch.no_grad()
def eval(env_name, env: gym.vector.VectorEnv, model: DecisionTransformer, rtg_target):
    # parallel evaluation with vectorized environment
    model.eval()
    
    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=bool)

    max_timestep = model.max_timestep
    context_len = model.context_len
    timesteps = torch.arange(max_timestep, device=device)
    state, _ = env.reset()
    
    states = TorchDeque(maxlen=context_len, device=device, dtype=torch.float32)
    actions = torch.zeros((episodes, max_timestep), dtype=torch.long, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep = rtg_target, 0

    while not done_flags.all() and timestep < model.max_timestep:
        states.append(torch.from_numpy(state))
        rewards_to_go[:, timestep] = reward_to_go - torch.from_numpy(returns).to(device).unsqueeze(-1)
        obs_index = torch.arange(max(0, timestep-context_len+1), timestep+1)
        action_preds = model.forward(states.to_tensor(),
                                        actions[:, obs_index],
                                        rewards_to_go[:, obs_index], # drop rewards
                                        timesteps[None, obs_index],
                                        ) 
        action = action_preds[:, -1].argmax(dim=-1).detach()
        actions[:, timestep] = action
        state, reward, dones, truncs, _ = env.step(action.cpu().numpy())
        returns += reward * ~done_flags
        done_flags = np.bitwise_or(np.bitwise_or(done_flags, dones), truncs)
        timestep += 1
        normalized_returns = get_normalized_score(env_name, returns)
    return np.mean(returns), np.std(returns), np.mean(normalized_returns), np.std(normalized_returns)

def train(cfg, seed, log_dict, idx, logger):
    log_to_wandb = cfg.log_to_wandb
    if log_to_wandb:
        mlp_description = "mlp_" if cfg.model.mlp_embedding else ""
        lora_description = f"lora_dim={cfg.model.lora_cfg.lora_attn_dim}_" if cfg.model.lora_cfg.use_lora else ""
        cotraining_description = f"cotrain={cfg.nlp_train.co_lambda}_" if cfg.nlp_train.co_training else ""
        pretrain_description = "_woPT" if (cfg.model.random_initialize or cfg.pretrained_lm == "none") else ""
        lr_description = f"lr={cfg.train.lr}_"
        wd_description = f"wd={cfg.train.weight_decay}_"
        wandb.init(
            name=f"{seed}-{np.random.randint(1000000)}",
            group=f"{cfg.env.env_name}-{cfg.dataset_type}-{cfg.model_name}-{cfg.pretrained_lm}{pretrain_description}-ratio={cfg.sample_ratio}-{mlp_description}{lora_description}{cotraining_description}{lr_description}{wd_description}{cfg.description}",
            entity="human-dex",
            project="wikiRL",
        )
    utils.config_logging("main.log")
    env_name = cfg.env.env_name
    env = gym.make(f'{env_name.lower()}-{cfg.buffer.dataset_type}-v0', stack=cfg.buffer.stack_frame)
    data_env = gym.make(f'{env_name.lower()}-{cfg.buffer.dataset_type}-v0', stack=False)
    eval_env = SyncVectorEnv([lambda: deepcopy(env) for _ in range(cfg.train.eval_episodes)])
    utils.set_seed_everywhere(eval_env, seed)

    state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
    print("initializing buffer")
    buffer = instantiate(cfg.buffer, env=data_env, seed=seed)
    model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim, action_space=eval_env.action_space[0], device=device)
    train_cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = CustomLRScheduler(optimizer, train_cfg.warmup_tokens, train_cfg.final_tokens)
    logger.info(f"Training seed {seed} for {train_cfg.train_steps} timesteps with {env_name} {buffer.dataset_type.title()} dataset")

    if cfg.nlp_train.co_training:
        if cfg.pretrained_lm == "none" or cfg.pretrained_lm.startswith("vit") or not cfg.pretrained_lm.startswith("gpt2"):
            raise ValueError("co training requires pretrained language model")
        print("co training with lambda="+str(cfg.nlp_train.co_lambda))
    
    if cfg.pretrained_lm.startswith("gpt2"):
        tokenizer_name = "gpt2"
        block_size=None
        
    if cfg.pretrained_lm != "none" and not cfg.pretrained_lm.startswith("vit") and cfg.pretrained_lm.startswith("gpt2") and cfg.nlp_train.co_training:
        train_nlp_dataloader, _ = get_dataset(
            dataset_name = cfg.nlp_train.nlp_dataset_name,
            dataset_config_name = cfg.nlp_train.nlp_dataset_config_name,
            tokenizer_name=tokenizer_name,
            block_size=block_size,
        )
        train_nlp_dataset = cycle(iter(train_nlp_dataloader))
    
    local_log_dict = log_dict
    for key in local_log_dict.keys():
        local_log_dict[key].append([])

    best_reward = -np.inf
    tokens = 0
    utils.write_to_dict(local_log_dict, 'rtg_target', train_cfg.rtg_target)
    
    outputs = dict()
    action_losses = []
    if cfg.pretrained_lm != "none" and not cfg.pretrained_lm.startswith("vit") and cfg.pretrained_lm.startswith("gpt2") and cfg.nlp_train.co_training:
        nlp_train_losses = []
    
    progress_bar = tqdm(range(1, train_cfg.train_steps + 1))
    
    for timestep in progress_bar:
        states, actions, rewards_to_go, timesteps, mask = buffer.sample(train_cfg.batch_size)
        # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
        action_logits = model.forward(states, actions, rewards_to_go, timesteps)
        action_logits = action_logits[mask]
        action_loss = F.cross_entropy(action_logits, actions[mask].detach().to(dtype=torch.long))
        utils.write_to_dict(local_log_dict, 'action_loss', action_loss.item())
        
        if cfg.pretrained_lm != "none" and not cfg.pretrained_lm.startswith("vit") and cfg.pretrained_lm.startswith("gpt2") and cfg.nlp_train.co_training:
            batch = next(train_nlp_dataset)
            lm_out = model.transformer(**batch)
            lm_loss = lm_out.loss
            nlp_train_losses.append(lm_loss.detach().cpu().item())
        if cfg.nlp_train.co_training:
            loss = action_loss + cfg.nlp_train.co_lambda * lm_loss
        else: 
            loss = action_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += (actions > 0).sum().item()
        scheduler.step(tokens)
        action_losses.append(action_loss.detach().cpu().item())
        if cfg.nlp_train.co_training and cfg.pretrained_lm != "none" and not cfg.pretrained_lm.startswith("vit") and cfg.pretrained_lm.startswith("gpt2") and cfg.nlp_train.co_training:
            progress_bar.set_postfix({"loss": action_losses[-1], "lm_loss": nlp_train_losses[-1], "lr": scheduler.get_lr()[0]})
        else: 
            progress_bar.set_postfix({"loss": action_losses[-1], "lr": scheduler.get_lr()[0]})
        if timestep % train_cfg.eval_interval == 0:
            if type(train_cfg.rtg_target) == list:
                for target in train_cfg.rtg_target:
                    eval_mean, eval_std, normalized_mean, normalized_std = eval(env_name, eval_env, model, target)
                    utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1)
                    utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean)
                    logger.info(f"Seed: {seed}, Step: {timestep}, Target: {target}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
                    outputs[f"evalutation/target_{target}_return_mean"] = eval_mean
                    outputs[f"evalutation/target_{target}_return_std"] = eval_std
                    outputs[f"evalutation/target_{target}_normalized_return_mean"] = normalized_mean
                    outputs[f"evalutation/target_{target}_normalized_return_std"] = normalized_std
                    if eval_mean > best_reward:
                        best_reward = eval_mean
                        model.save(f'best_train_seed_{seed}' if timestep <= train_cfg.train_steps else f'best_finetune_seed_{seed}')
                        logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {timestep} with rtg target {target}')
                    
            else:
                eval_mean, eval_std, normalized_mean, normalized_std = eval(env_name, eval_env, model, train_cfg.rtg_target)
                utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1)
                utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean)
                logger.info(f"Seed: {seed}, Step: {timestep}, Target: {train_cfg.rtg_target}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
                outputs[f"evalutation/target_{train_cfg.rtg_target}_return_mean"] = eval_mean
                outputs[f"evalutation/target_{train_cfg.rtg_target}_return_std"] = eval_std
                outputs[f"evalutation/target_{train_cfg.rtg_target}_normalized_return_mean"] = normalized_mean
                outputs[f"evalutation/target_{train_cfg.rtg_target}_normalized_return_std"] = normalized_std
                if eval_mean > best_reward:
                    best_reward = eval_mean
                    model.save(f'best_train_seed_{seed}' if timestep <= train_cfg.train_steps else f'best_finetune_seed_{seed}')
                    logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {timestep} with rtg target {train_cfg.rtg_target}')
            
            outputs[f"training/loss_mean"] = np.mean(action_losses)
            outputs[f"training/loss_std"] = np.std(action_losses)
            if cfg.pretrained_lm != "none" and not cfg.pretrained_lm.startswith("vit") and cfg.pretrained_lm.startswith("gpt2") and cfg.nlp_train.co_training:
                outputs[f"training/lm_loss_mean"] = np.mean(nlp_train_losses)
                outputs[f"training/lm_loss_std"] = np.std(nlp_train_losses)
            if log_to_wandb:
                wandb.log(outputs)
            
            action_losses = []
            if cfg.pretrained_lm != "none" and not cfg.pretrained_lm.startswith("vit") and cfg.pretrained_lm.startswith("gpt2") and cfg.nlp_train.co_training:
                nlp_train_losses = []
            
        if timestep == train_cfg.train_steps:
            model.save(f'final_train_seed_{seed}')
            model.load(f'best_train_seed_{seed}')

    logger.info(f"Finish training seed {seed} with average eval mean: {eval_mean}")
    eval_env.close()
    return eval_mean