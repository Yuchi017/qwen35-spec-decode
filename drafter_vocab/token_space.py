"""How big does the drafter's token space (lm_head vocabulary) need to be?

Counts the tokens the drafter has to predict -- the supervised (loss_mask=1) positions of the
prepared training data in ./out -- ranks them by train frequency, and reports for a range of
vocabulary sizes N:
  - coverage: share of supervised tokens inside the top-N set, on train and on the official
    validation split (ranking from train only, so val shows how well it generalises)
  - chain coverage: share of length-k windows whose tokens are all inside the set (a drafter
    proposing k tokens can only get all k right if all of them are in its vocabulary)
  - lm_head size relative to the full vocabulary

Writes the train ranking to token_space/token_freq_train.npy (token id -> count) and
token_space/top_ids_train.npy (token ids sorted by count) for building a reduced lm_head later.

  python token_space.py                       # ./out, same train/val split as training
  python token_space.py --sizes 4096 16384 32768 --chain 5 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

TRAIN_RATIO = 0.978203  # train_mtp.sh / train_dflash.sh: int(28742 * ratio) == 28115 train rows


def count(ds, vocab_size, chain_ks):
    """Token counts at supervised positions (and over all positions) plus the supervised
    token sequences per row (needed for chain coverage)."""
    sup = np.zeros(vocab_size, dtype=np.int64)
    allc = np.zeros(vocab_size, dtype=np.int64)
    seqs = []
    for batch in ds.iter(batch_size=256):
        for ids, mask in zip(batch["input_ids"], batch["loss_mask"]):
            ids = np.asarray(ids, dtype=np.int64)
            mask = np.asarray(mask, dtype=bool)
            allc += np.bincount(ids, minlength=vocab_size)
            s = ids[mask]
            sup += np.bincount(s, minlength=vocab_size)
            if chain_ks:
                # keep contiguous supervised spans separate so windows never cross a turn boundary
                edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
                seqs.extend(ids[a:b] for a, b in zip(edges[::2], edges[1::2]))
    return sup, allc, seqs


def chain_coverage(seqs, in_set, k):
    """Share of length-k windows (inside supervised spans) whose tokens are all in the set."""
    total = ok = 0
    for s in seqs:
        if len(s) < k:
            continue
        m = in_set[s].astype(np.int32)
        run = np.convolve(m, np.ones(k, dtype=np.int32), mode="valid")
        total += len(run)
        ok += int((run == k).sum())
    return ok / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./out")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--chain", type=int, nargs="*", default=[5, 8], help="window lengths k")
    ap.add_argument("--hidden", type=int, default=2560, help="lm_head input size")
    ap.add_argument("--out-dir", default="token_space")
    ap.add_argument("--top", type=int, default=30, help="print the N most frequent tokens")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    ds = load_from_disk(a.data).select_columns(["input_ids", "loss_mask"]).with_format("numpy")
    n_train = int(len(ds) * TRAIN_RATIO)
    train, val = ds.select(range(n_train)), ds.select(range(n_train, len(ds)))
    vocab = max(len(tok), 248320)
    print(f"rows: train {len(train)}, val {len(val)}; vocab (lm_head rows) {vocab}")

    sup_tr, all_tr, seq_tr = count(train, vocab, a.chain)
    sup_va, _, seq_va = count(val, vocab, a.chain)
    order = np.argsort(-sup_tr, kind="stable")
    rank = np.empty(vocab, dtype=np.int64)
    rank[order] = np.arange(vocab)

    n_sup_tr, n_sup_va = sup_tr.sum(), sup_va.sum()
    print(f"supervised tokens: train {n_sup_tr:,}  val {n_sup_va:,}  "
          f"(all tokens in train rows {all_tr.sum():,})")
    print(f"distinct supervised token ids: train {(sup_tr > 0).sum():,}  val {(sup_va > 0).sum():,}  "
          f"val ids never supervised in train {((sup_va > 0) & (sup_tr == 0)).sum():,} "
          f"({sup_va[(sup_tr == 0)].sum() / n_sup_va:.4%} of val tokens)")
    print(f"distinct ids anywhere in train (prompt + output): {(all_tr > 0).sum():,}")
    for p in (0.9, 0.95, 0.99, 0.995, 0.999, 0.9999):
        n = int(np.searchsorted(np.cumsum(sup_tr[order]) / n_sup_tr, p) + 1)
        print(f"  top-N covering {p:.2%} of train supervised tokens: N = {n:,}")

    print(f"\n{'N':>7} | {'train cov':>9} | {'val cov':>8} | "
          + " | ".join(f"{'val chain' + str(k):>11}" for k in a.chain)
          + f" | {'lm_head':>8} | {'vs full':>7}")
    rows = []
    for n in a.sizes + [vocab]:
        in_set = rank < n
        tr = sup_tr[in_set].sum() / n_sup_tr
        va = sup_va[in_set].sum() / n_sup_va
        ch = [chain_coverage(seq_va, in_set, k) for k in a.chain]
        mb = n * a.hidden * 2 / 2**20  # bf16 weights
        rows.append({"N": n, "train_cov": tr, "val_cov": va,
                     **{f"val_chain{k}": c for k, c in zip(a.chain, ch)}, "lm_head_MiB": mb})
        print(f"{n:>7} | {tr:>9.4%} | {va:>8.4%} | " + " | ".join(f"{c:>11.4%}" for c in ch)
              + f" | {mb:>6.0f}MB | {n / vocab:>7.2%}")

    print(f"\nmost frequent supervised tokens (train):")
    for i in order[: a.top]:
        print(f"  {i:>7}  {sup_tr[i] / n_sup_tr:7.3%}  {tok.convert_ids_to_tokens(int(i))!r}")

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    np.save(out / "token_freq_train.npy", sup_tr)
    np.save(out / "top_ids_train.npy", order[(sup_tr[order] > 0)])
    (out / "coverage.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}/token_freq_train.npy, top_ids_train.npy, coverage.json")


if __name__ == "__main__":
    main()
