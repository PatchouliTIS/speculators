from transformers import AutoConfig
c = AutoConfig.from_pretrained("/tmp/ray-app/qwen36_new", trust_remote_code=True)
t = getattr(c, "thinker_config", c); t = getattr(t, "text_config", t)
print("hidden_size:", t.hidden_size,
    "num_attention_heads:", t.num_attention_heads,
    "num_key_value_heads:", t.num_key_value_heads,
    "head_dim:", getattr(t, "head_dim", "N/A"))