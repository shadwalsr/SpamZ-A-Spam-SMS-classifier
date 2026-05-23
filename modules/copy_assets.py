import os, shutil

src_dir = os.path.join('models', 'distilbert-spam', 'best')
dst_dir = os.path.join('models', 'distilbert-spam')

os.makedirs(dst_dir, exist_ok=True)
files_to_copy = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json', 'model.onnx']
for fname in files_to_copy:
    src_path = os.path.join(src_dir, fname)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_dir)
        print(f"Copied {fname} to {dst_dir}")
    else:
        print(f"{fname} not found in {src_dir}")
