import json, pathlib
p = pathlib.Path("/apdcephfs_fsgm/share_303700817/patchychen/ImageCaption/Eagle3Draft/0603_v2ft5-1_best/config.json")
c = json.loads(p.read_text())
tlc = c["transformer_layer_config"]
print("partial_rotary_factor:", tlc["rope_parameters"].get("partial_rotary_factor"))
print("mrope_section       :", tlc["rope_parameters"].get("mrope_section"))
print("head_dim            :", tlc["head_dim"])
print("→ rotary_dim (vLLM) :", int(tlc["head_dim"] * tlc["rope_parameters"].get("partial_rotary_factor", 1.0)))