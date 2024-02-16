from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
import numpy as np
def get_custom_gpt2_model(model_type='random_pretrain'):
    if model_type == 'random_pretrain':
        path = '/home/yuyao/meta-rl/atari/random_pretrain.pt'
    elif model_type == 'bad_pretrain':
        path = '/home/yuyao/meta-rl/atari/bad_pretrain.pt'
    else:
        raise ValueError('model_type must be either random_pretrain or bad_pretrain')
    checkpoint = torch.load(path)
    new_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('transformer_model.'):
            # delete the prefix
            new_key = key[18:]
            new_dict[new_key] = value
    config = GPT2Config.from_pretrained('gpt2')
    lm_model = GPT2LMHeadModel(config)
    lm_model.load_state_dict(new_dict, strict=False)
    return lm_model

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    lm_model = get_custom_gpt2_model('random_pretrain')
    print(lm_model)

    # # generate a sequence of tokens
    lm_model.to('cuda')
    lm_model.eval()
    input_ids = torch.tensor(tokenizer.encode("Hello, my name is Tom. I come from", add_special_tokens=True)).unsqueeze(0).to('cuda')  # Batch size 1
    outputs = lm_model.generate(input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

