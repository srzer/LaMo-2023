# Unleashing the Power of Pre-trained Language Models for Offline Reinforcement Learning

Ruizhe Shi*, Yuyao Liu*, Yanjie Ze, Simon S. Du, Huazhe Xu

[Project Page](https://lamo2023.github.io)

## Experiments on D4RL

### Environment
You need to install 

1. packages in ``experiment-atari/env.yml`` 
2. D4RL (follow the guidance in [D4RL](https://github.com/Farama-Foundation/D4RL))

### Data
To download original D4RL data, 
```bash
cd data
python download_d4rl_datasets.py
```

To get downsampled data, you need to modify line 10 of 'data/mujoco/ratio_dataset.py' and line 10 of 'data/kitchen/ratio_dataset.py' as 
```python
suffix = [your data version name]
```
and then run
```bash
cd data
cd mujoco
python ratio_dataset.py
cd ..
cd kitchen
python ratio_dataset.py
cd ..
```

Besides, you can directly get our pre-processed data in [this link](https://drive.google.com/drive/folders/1c3htmB0bCixakM12EmDG4Qr3nMihrj6t?usp=sharing).

### Tasks
We provide 8 Tasks in total, of various data ratios:

- D4RL
  - MuJoCo: Hopper, Walker2d, HalfCheetah, Reacher2d
  - Kitchen
- Atari: Breakout, Qbert, Pong (code coming soon)

### Usage

After installing the packages and data, to reproduce our results on D4RL, you only need to run
```bash
cd code
bash scripts.sh [env_name] [dataset_name] [sample_ratio] [description] [seed] [gpu]
```

An example is:
```bash
bash scripts.sh hopper medium 0.1 reproduce 0 0
```

If you want to view results on [Weights & Biases](wandb.ai), you need to modify line 435, 436 of '/code/experiment.py' as:
```python
entity=[your-group-name],
project=[your-project-name],
```

Trying more configurations is encouraged! Important arguments are explained as below:

```bash
-w # enable wandb
--sample_ratio your_sample_ratio # determine the size of the data you are training on, like 0.1
--data_suffix your_data_version_name # you could downsample the data by yourself, default is "d1"
--mlp_embedding # use MLP as embeddings and projections
--adapt_mode # otherwise fully fine-tuning
--adapt_embed # fine-tune embeddings and projections when adapt_mode is ON
--lora # fine-tune low rank matrices of Transformer when adapt_mode is ON
--pretrained_lm language_model_name # you could try 'gpt2' and 'gpt2-medium'
--co_training # use language loss as auxiliary objective
--co_lambda # the weight of language loss, like 0.1
```

## Experiments on Atari

### Environment
You need to install 

1. packages in env.yml 
2. D4RL (follow the guidance in [D4RL](https://github.com/Farama-Foundation/D4RL))

### Data
To download original D4RL data, 
```bash
cd data
python download_d4rl_datasets.py
```

To get downsampled data, you need to modify line 10 of 'data/mujoco/ratio_dataset.py' and line 10 of 'data/kitchen/ratio_dataset.py' as 
```python
suffix = [your data version name]
```
and then run
```bash
cd data
cd mujoco
python ratio_dataset.py
cd ..
cd kitchen
python ratio_dataset.py
cd ..
```

Besides, you can directly get our pre-processed data in [this link](https://drive.google.com/drive/folders/1c3htmB0bCixakM12EmDG4Qr3nMihrj6t?usp=sharing).

### Tasks
We provide 8 Tasks in total, of various data ratios:

- D4RL
  - MuJoCo: Hopper, Walker2d, HalfCheetah, Reacher2d
  - Kitchen
- Atari: Breakout, Qbert, Pong (code coming soon)

### Usage

After installing the packages and data, to reproduce our results on D4RL, you only need to run
```bash
cd code
bash scripts.sh [env_name] [dataset_name] [sample_ratio] [description] [seed] [gpu]
```

An example is:
```bash
bash scripts.sh hopper medium 0.1 reproduce 0 0
```

If you want to view results on [Weights & Biases](wandb.ai), you need to modify line 435, 436 of '/code/experiment.py' as:
```python
entity=[your-group-name],
project=[your-project-name],
```

Trying more configurations is encouraged! Important arguments are explained as below:

```bash
-w # enable wandb
--sample_ratio your_sample_ratio # determine the size of the data you are training on, like 0.1
--data_suffix your_data_version_name # you could downsample the data by yourself, default is "d1"
--mlp_embedding # use MLP as embeddings and projections
--adapt_mode # otherwise fully fine-tuning
--adapt_embed # fine-tune embeddings and projections when adapt_mode is ON
--lora # fine-tune low rank matrices of Transformer when adapt_mode is ON
--pretrained_lm language_model_name # you could try 'gpt2' and 'gpt2-medium'
--co_training # use language loss as auxiliary objective
--co_lambda # the weight of language loss, like 0.1
```

## Acknowledgement
Our work is based on many open-source projects, including [Decision Transformer](https://github.com/kzl/decision-transformer), [Can Wikipedia Help Offline Reinforcement Learning](https://github.com/machelreid/can-wikipedia-help-offline-rl), [LoRA](https://github.com/microsoft/LoRA). We thank all these authors for their nicely open sourced code and their great contributions to the community.