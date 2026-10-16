"""Download the ChronoGPT-Instruct vintages 2010 to 2023 and the GPT-2 tokenizer into $ECALLS_LLM_CACHE."""
import os, time
from huggingface_hub import hf_hub_download
CACHE = os.environ.get("ECALLS_LLM_CACHE", ".llm-cache")
os.makedirs(CACHE, exist_ok=True)
for f in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "config.json"):
    hf_hub_download("openai-community/gpt2", f, cache_dir=CACHE)
print("gpt2 tokenizer ok", flush=True)
for y in range(2010, 2024):
    repo = f"manelalab/chrono-gpt-instruct-v1-{y}1231"
    t = time.time()
    for f in ("config.pt", "config.json", "pytorch_model.bin"):
        hf_hub_download(repo, f, cache_dir=CACHE)
    print(repo, "ok", round(time.time() - t), "s", flush=True)
print("ALL DONE", flush=True)
