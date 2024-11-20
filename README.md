# Flash attention implementation

`kernel.py` implements the flash_attn algorithm defined as the original paper

`kernel_v2.py` is an alternative version same as https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py

`patch.py` replaces the torch scaled_dot_product_attention with kernel functions
