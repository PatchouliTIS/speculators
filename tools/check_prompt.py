from datasets import load_from_disk
ds = load_from_disk("/deploy/qwen36_dflash/data/qwen36_new_imagecaption_v5-1").with_format(None)
# 检查前 5 条多模态样本的 messages_json 是否非空
for i in range(min(5, len(ds))):
    mj = ds[i].get("messages_json", "")
    if mj:
        import json
        msgs = json.loads(mj)
        print(f"Sample {i}: has {sum(1 for t in msgs for s in (t.get('content') or []) if isinstance(s, dict) and 'image_url' in s.get('type',''))} image refs")
        break
else:
    print("WARNING: No sample has messages_json — multi_modal_data will NOT be sent to vLLM!")