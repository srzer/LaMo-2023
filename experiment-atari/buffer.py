import torch
import numpy as np
from dotmap import DotMap
import d4rl_atari
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AtariBuffer:
    def __init__(self, env, dataset_type, context_len, stack_frame=4, sample_type='traj_length', seed=0, sample_ratio=1) -> None:
        self.dataset_type = dataset_type
        if dataset_type in ['medium', 'expert', 'mixed']:
            # when getting the dataset, we don't want to stack, and we will stack the frames ourselves.
            self.dataset = DotMap(env.get_dataset())
        else: 
            raise NotImplementedError
        self.rng = np.random.default_rng(seed)
        self.dataset.terminals = self.dataset.terminals.astype(bool)
        self.dataset.terminals[-1] = True
        self.traj_sp = np.insert(np.where(self.dataset.terminals)[0][:-1]+1, 0, 0) # start is inclusive
        self.traj_ep = np.where(self.dataset.terminals)[0] # end is also inclusive 
        self.traj_length = self.traj_ep - self.traj_sp + 1
        if self.traj_length[-1] < context_len: # pad to avoid index out of bound when fetching the last trajecectory, this is safe as we mask out padded steps.
            padding_length = context_len - self.traj_length[-1]
            self.dataset.observations = np.concatenate((self.dataset.observations, np.zeros((padding_length, 1, 84, 84), dtype=self.dataset.observations.dtype)), axis=0)
        self.traj_returns = np.add.reduceat(self.dataset.rewards, self.traj_sp)
        self.num_trajs = len(self.traj_sp)
        self.rewards_to_go = np.cumsum(self.traj_returns)[np.insert(np.cumsum(self.dataset.terminals), 0, 0)[:-1]] - np.cumsum(self.dataset.rewards)
        self.p_sample = np.ones(self.num_trajs) / self.num_trajs if sample_type == 'uniform' else self.traj_returns / \
            self.traj_returns.sum() if sample_type == 'traj_return' else self.traj_length / self.traj_length.sum()
        self.context_len = context_len
        self.stack_frame = stack_frame
        self.size = self.dataset.rewards.shape[0] - self.context_len * self.num_trajs
        print(f"using dataset {dataset_type} with {self.num_trajs} trajectories ({self.dataset.rewards.shape[0]} transitions), average return: {np.mean(self.traj_returns)}, variance: {np.var(self.traj_returns)}")
        assert(sample_ratio > 0 and sample_ratio <= 1)
        if sample_ratio < 1:
            self.ratio_dataset(sample_ratio)
        print(self.dataset.observations.shape)
        self.observations_tensor = torch.from_numpy(self.dataset.observations).squeeze(1).to(dtype=torch.float32, device=device)
        self.actions_tensor = torch.from_numpy(self.dataset.actions).to(dtype=torch.int32, device=device)
        self.rewards_to_go_tensor = torch.from_numpy(self.rewards_to_go).unsqueeze(-1).to(dtype=torch.float32, device=device)

    def ratio_dataset(self, sample_ratio):
        sample_indices = self.rng.choice(self.num_trajs, size=int(np.floor(self.num_trajs*sample_ratio)), p=self.p_sample, replace=False)
        sample_indices.sort()
        self.num_trajs = len(sample_indices)
        self.p_sample = self.p_sample[sample_indices]/np.sum(self.p_sample[sample_indices])
        self.traj_sp = self.traj_sp[sample_indices]
        self.traj_ep = self.traj_ep[sample_indices]
        self.traj_length = self.traj_length[sample_indices]
        print(f"sample {sample_ratio} of dataset, {len(sample_indices)} trajectories ({np.sum(self.traj_length)} transitions) after sampling, average return: {np.mean(self.traj_returns[sample_indices])}")
        
    def sample(self, batch_size):
        selected_traj = self.rng.choice(np.arange(self.num_trajs), batch_size, replace=True, p=self.p_sample)
        selected_traj_sp = self.traj_sp[selected_traj]
        selected_offset = np.floor(self.rng.random(batch_size) * (self.traj_length[selected_traj] - self.context_len)).astype(np.int32).clip(min=0)
        selected_sp = selected_traj_sp + selected_offset
        selected_ep = (selected_sp + self.context_len).clip(max=self.traj_ep[selected_traj])
        # fill the index of those padded steps with -1, so that we can fetch the last step of the corresponding item
        selected_index = selected_sp[:, None] + np.arange(self.context_len)
        selected_index = np.where(selected_index <= selected_ep[:, None], selected_index, -1)
        masks = selected_index >= 0
        masks = torch.from_numpy(masks).to(dtype=torch.bool, device=device)
        timesteps = selected_offset[:, None] + np.arange(self.context_len)  # we don't care about the timestep for those padded steps
        states_list = []
        for traj_sp, sp, offset in zip(selected_traj_sp, selected_sp, selected_offset):
            if offset >= self.stack_frame: # don't need padding
                observation_list = []
                for i in range(self.context_len):
                    observation_list.append(self.observations_tensor[sp+i-self.stack_frame+1:sp+i+1]) # stack_frame * 84 * 84
                states_list.append(torch.stack(observation_list))
            else:
                observation_list = []
                for i in range(self.context_len):
                    if self.stack_frame - offset - i - 1 >= 0: # need padding
                        padding_length = self.stack_frame - offset - i - 1
                        padding_tensor = torch.zeros((padding_length, self.observations_tensor.shape[1], self.observations_tensor.shape[2]), dtype=self.observations_tensor.dtype).to(device)
                        observation_list.append(torch.cat((padding_tensor, self.observations_tensor[traj_sp:sp+i+1]), dim=0))
                    else:
                        observation_list.append(self.observations_tensor[sp+i-self.stack_frame+1:sp+i+1])
                states_list.append(torch.stack(observation_list))
                
        states = torch.stack(states_list)
        selected_index = torch.tensor(selected_index).to(dtype=torch.int64, device=device)
        actions = self.actions_tensor[selected_index]
        rewards_to_go = self.rewards_to_go_tensor[selected_index]
        timesteps = torch.as_tensor(timesteps).to(dtype=torch.int32, device=device)
        return states, actions, rewards_to_go, timesteps, masks