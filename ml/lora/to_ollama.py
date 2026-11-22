"""
Package the merged LoRA model for Ollama as `dentagent-clinical`.

    python ml/lora/train.py --merge      # first: ml/lora/out/merged (safetensors)
    python ml/lora/to_ollama.py          # then: ollama create dentagent-clinical

The chat template and stop tokens are copied from the installed base model
(`ollama show qwen3:1.7b --modelfile`), so the fine-tuned model is prompted exactly
like the base one — including think=False handling. Quantised to q4_K_M like the
base tag (Ollama >= 0.40: "int4"; set LORA_QUANT=q4_K_M on older versions).
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(HERE, "out", "merged")
NAME = os.getenv("LORA_MODEL", "dentagent-clinical")
BASE_TAG = os.getenv("LORA_BASE_TAG", "qwen3:1.7b")
# 4-bit like the base tag. Ollama >= 0.40 names it "int4" (older versions: "q4_K_M")
QUANT = os.getenv("LORA_QUANT", "int4")


def main():
    if not os.path.isdir(MERGED):
        sys.exit(f"{MERGED} not found — run: python ml/lora/train.py --merge")
    base = subprocess.run(["ollama", "show", BASE_TAG, "--modelfile"], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout
    kept = [ln for ln in base.splitlines() if not ln.startswith(("FROM ", "#"))]
    modelfile = f"FROM {MERGED}\n" + "\n".join(kept) + "\n"
    path = os.path.join(HERE, "out", "Modelfile")
    with open(path, "w", encoding="utf-8") as f:
        f.write(modelfile)
    print(f"wrote {path}; creating {NAME} ...", flush=True)
    subprocess.run(["ollama", "create", NAME, "-f", path, "--quantize", QUANT], check=True)
    print(f"done: ollama run {NAME}")


if __name__ == "__main__":
    main()
