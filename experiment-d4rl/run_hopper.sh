export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

model_type=dt
lr=1e-4
lmlr=1e-5
weight_decay=1e-5
dropout=0.1
warmup_steps=2500 
num_steps_per_iter=2500 
max_iters=40 
num_eval_episodes=20

env=${1}
if [ "$env" == "reacher2d" ]; then
    K=5
else
    K=20
fi # K is context length
dataset=${2}
sample_ratio=${3}
pretrained_lm="gpt2"
description=${4}
seed=${5}
description="${pretrained_lm}_pretrained-ratio=${sample_ratio}_${description}"
gpu=${6}
outdir="checkpoints/${env}_${dataset}_${description}_${seed}"

CUDA_VISIBLE_DEVICES=${gpu} python experiment.py --env ${env} \
        --dataset ${dataset} \
        --model_type ${model_type} \
        --seed ${seed} \
        --K ${K} \
        -lr ${lr} \
        -lmlr ${lmlr} \
        --num_steps_per_iter ${num_steps_per_iter} \
        --weight_decay ${weight_decay} \
        --max_iters ${max_iters} \
        --num_eval_episodes ${num_eval_episodes} \
        --sample_ratio ${sample_ratio} \
        --warmup_steps ${warmup_steps} \
        --pretrained_lm ${pretrained_lm} \
        --adapt_mode \
        --adapt_embed \
        --lora \
        --outdir ${outdir} \
        --dropout ${dropout} \
        --description ${description} \
        -w