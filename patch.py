import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time

from kernel_debug import flash_attn_forward
from transformers import AutoTokenizer, AutoModelForCausalLM

def flash_attn(query, key, value, attn_mask=None, dropout_p=0.0, 
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    return flash_attn_forward(query, key, value, softmax_scale=scale, causal=is_causal)

F.scaled_dot_product_attention = flash_attn

model_path = "/data/private/lizhenyu/llama-2-7b-chat"
model = AutoModelForCausalLM.from_pretrained(model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    device_map='auto',
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_text = "Who does \"You-Know-Who\" refer to in Harry Potter?\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

output = model.generate(
    input_ids,
    max_length=400,
    num_beams=1,
    do_sample=False
)
print(tokenizer.decode(output.cpu()[0]))
