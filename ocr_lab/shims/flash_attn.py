def flash_attn_varlen_func(*args, **kwargs):
    raise RuntimeError("flash_attn CPU shim was called unexpectedly")
