"""Greedy two-round run for exact-match comparison between servers.

Round 1: N Spec-Bench prompts (chat template rendered by the server's /tokenize).
Round 2: round-1 prompt + the server's own round-1 output + a follow-up user turn,
so the prompt prefix covers blocks whose Mamba state was written during decode.
Requests go one at a time so batching never differs between runs.

  python exact_match.py run <name> [N]           # two greedy rounds against the server on :8001
  python exact_match.py cmp <name_a> <name_b>     # identical outputs / first divergence position
  python exact_match.py lp <name> [N]            # same, with top-5 logprobs at every position
  python exact_match.py margin <name_a> <name_b>  # a's top1-top2 logprob gap at each first divergence
  python exact_match.py selfhit <name> <N> a|b   # a: decode-written-block hit run, b: cold rerun

A kernel change that only reorders floating-point math flips greedy argmax at near-ties
(gap of a few bf16 ulps); a real bug diverges at confident positions too. Compare the
`margin` gaps with the gap distribution over all positions.
"""

import json
import sys
import urllib.request

BASE = "http://localhost:8001"
PROMPTS = "data/spec_bench_question.jsonl"
OUT = "results/exact_{}.json"
MODEL = "qwen35-4b"
FOLLOWUP = "<|im_end|>\n<|im_start|>user\nContinue from the answer above and summarize it in one paragraph.<|im_end|>\n<|im_start|>assistant\n"


def load_prompts(n):
    """First n Spec-Bench questions as single-turn chats."""
    rows = [json.loads(line) for line in open(PROMPTS)][:n]
    return [{"messages": [{"role": "user", "content": r["turns"][0]}]} for r in rows]


def post(path, body):
    r = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def complete(prompt_ids, max_tokens, logprobs=False, force_len=False):
    res = post(
        "/v1/completions",
        {
            **({"ignore_eos": True, "min_tokens": max_tokens} if force_len else {}),
            "model": MODEL,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "return_token_ids": True,
            "skip_special_tokens": False,
            **({"logprobs": 5} if logprobs else {}),
        },
    )
    choice = res["choices"][0]
    if logprobs:
        return choice["token_ids"], choice["logprobs"]["top_logprobs"]
    return choice["token_ids"]


def run(name, n):
    reqs = load_prompts(n)
    follow = post("/tokenize", {"model": MODEL, "prompt": FOLLOWUP,
                                "add_special_tokens": False})["tokens"]
    rows = []
    for i, req in enumerate(reqs):
        p1 = post("/tokenize", {"model": MODEL, "messages": req["messages"],
                                "tools": req.get("tools"), "add_generation_prompt": True})["tokens"]
        o1 = complete(p1, 600)
        p2 = p1 + o1 + follow
        o2 = complete(p2, 300)
        rows.append({"p1": len(p1), "o1": o1, "o2": o2})
        print(f"{i:3d} prompt {len(p1)} out1 {len(o1)} out2 {len(o2)}", flush=True)
    json.dump(rows, open(OUT.format(name), "w"))


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cmp(a, b):
    A, B = json.load(open(OUT.format(a))), json.load(open(OUT.format(b)))
    for rnd in ("o1", "o2"):
        same = 0
        diffs = []
        for x, y in zip(A, B):
            if rnd == "o2" and x["o1"] != y["o1"]:
                continue  # round-2 prompts differ; not comparable
            d = first_diff(x[rnd], y[rnd])
            if d is None:
                same += 1
            else:
                diffs.append(d)
        total = same + len(diffs)
        print(f"{a} vs {b} {rnd}: identical {same}/{total}  first-diff positions {sorted(diffs)}")


def run_lp(name, n):
    """Both rounds, with the top-5 logprobs at every generated position."""
    reqs = load_prompts(n)
    follow = post("/tokenize", {"model": MODEL, "prompt": FOLLOWUP,
                                "add_special_tokens": False})["tokens"]
    rows = []
    for req in reqs:
        p1 = post("/tokenize", {"model": MODEL, "messages": req["messages"],
                                "tools": req.get("tools"), "add_generation_prompt": True})["tokens"]
        o1, lp = complete(p1, 600, logprobs=True)
        o2, lp2 = complete(p1 + o1 + follow, 300, logprobs=True)
        rows.append({"o1": o1, "lp": lp, "o2": o2, "lp2": lp2})
    json.dump(rows, open(OUT.format(name), "w"))


def margins(a, b):
    """At each first divergence: a's top-1 minus top-2 logprob (how close to a tie a was)."""
    A, B = json.load(open(OUT.format(a))), json.load(open(OUT.format(b)))
    for o, lpk in (("o1", "lp"), ("o2", "lp2")):
        out = []
        for x, y in zip(A, B):
            if o not in x:
                continue
            if o == "o2" and x["o1"] != y["o1"]:
                continue
            d = first_diff(x[o], y[o])
            if d is None or d >= len(x[lpk]):
                continue
            top = sorted(x[lpk][d].values(), reverse=True)
            out.append(round(top[0] - top[1], 3) if len(top) > 1 else None)
        print(f"{o}: {a} top1-top2 logprob gap at first divergence vs {b}: {sorted(out)}")


def selfhit(name, n, phase):
    """phase a: 600 forced decode tokens (crosses >= 1 Mamba block), then the
    follow-up turn on the same server, so its prefix hits decode-written blocks.
    phase b (fresh server): the same follow-up prompts only, state from prefill."""
    path = OUT.format(f"selfhit_{name}")
    if phase == "a":
        reqs = load_prompts(n)
        follow = post("/tokenize", {"model": MODEL, "prompt": FOLLOWUP,
                                    "add_special_tokens": False})["tokens"]
        rows = []
        for req in reqs:
            p1 = post("/tokenize", {"model": MODEL, "messages": req["messages"],
                                    "tools": req.get("tools"), "add_generation_prompt": True})["tokens"]
            o1 = complete(p1, 600, force_len=True)
            p2 = p1 + o1 + follow
            o2, lp2 = complete(p2, 200, logprobs=True)
            rows.append({"p2": p2, "hit": o2, "hit_lp": lp2})
        json.dump(rows, open(path, "w"))
    else:
        rows = json.load(open(path))
        for r in rows:
            r["cold"], r["cold_lp"] = complete(r["p2"], 200, logprobs=True)
        json.dump(rows, open(path, "w"))
        gaps, same = [], 0
        for r in rows:
            d = first_diff(r["hit"], r["cold"])
            if d is None:
                same += 1
                continue
            top = sorted(r["cold_lp"][d].values(), reverse=True)
            gaps.append(round(top[0] - top[1], 3))
        print(f"{name}: hit-vs-cold identical {same}/{len(rows)}; cold top1-top2 gap at divergence {sorted(gaps)}")


if __name__ == "__main__":
    if sys.argv[1] == "selfhit":
        selfhit(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        sys.exit()
    if sys.argv[1] == "lp":
        run_lp(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    elif sys.argv[1] == "margin":
        margins(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    else:
        cmp(sys.argv[2], sys.argv[3])
