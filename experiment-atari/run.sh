# pretrained_lm can be none or gpt2
python main.py \
    env="breakout" \
    dataset_type="medium" \
    model_name="dt" \
    pretrained_lm="gpt" \
    description="desc" \
    seed=0 \
    sample_ratio=0.1 \
    log_to_wandb=0 \
    model.mlp_embedding=1 \
    model.random_initialize=0 \
    model.adapt_cfg.use_adapt=1 \
    model.adapt_cfg.adapt_embed=1 \
    model.lora_cfg.use_lora=1 \
    model.lora_cfg.lora_attn_dim=32 \
    model.context_len=30 \
    train.lr=1e-3 \
    train.weight_decay=0.01 \
    train.batch_size=128 \
    nlp_train.co_training=0 \
    nlp_train.co_lambda=0.1 \
