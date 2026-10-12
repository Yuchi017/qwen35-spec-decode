"""8,000 Stanford Alpaca instructions (seed 0) as prompts for response regeneration."""

import json
import random
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
Path("data").mkdir(exist_ok=True)
rows = json.loads(urllib.request.urlopen(URL).read())
random.Random(0).shuffle(rows)
with open("data/alpaca_prompts_8k.jsonl", "w") as f:
    for i, r in enumerate(rows[:8000]):
        prompt = r["instruction"].strip()
        if r.get("input", "").strip():
            prompt += "\n\n" + r["input"].strip()
        f.write(json.dumps({"id": f"alpaca-{i}", "prompt": prompt}, ensure_ascii=False) + "\n")
