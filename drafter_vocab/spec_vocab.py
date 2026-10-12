"""Reduced draft vocabulary for greedy drafting on vLLM's V2 model runner (MTP, DFlash, ...).

SPEC_DRAFT_VOCAB=<top_ids.npy>:<N>   e.g. token_space/top_ids_train.npy:65536

After the drafter is loaded, the rows of its lm_head for the N most frequent target tokens
(token_space.py ranking) are copied into a small [N, hidden] matrix. Greedy draft
sampling then computes logits over those N tokens only, takes the argmax and maps it back to the
full-vocab id, so the drafter never touches the full 248k-row lm_head. Verification still uses the
target's full vocabulary, so outputs are unchanged; tokens outside the set are just never drafted.
Probabilistic draft sampling and use_local_argmax_reduction keep the original path.
"""

import os

import numpy as np
import torch

_SPEC = os.environ.get("SPEC_DRAFT_VOCAB", "")


def _patch_speculator(mod):
    cls = mod.DraftModelSpeculator
    orig_load = cls.load_model
    orig_sample = cls.sample_draft

    def load_model(self, *args, **kwargs):
        orig_load(self, *args, **kwargs)
        path, _, n = _SPEC.rpartition(":")
        weight = self.model.lm_head.weight
        ids = np.load(path)[: int(n)]
        ids = torch.as_tensor(ids[ids < weight.shape[0]], dtype=torch.int64, device=weight.device)
        self._vocab_ids = ids
        self._vocab_weight = weight.index_select(0, ids).contiguous()
        print(f"[spec_vocab] {type(self.model).__name__} draft lm_head {tuple(weight.shape)} -> "
              f"{tuple(self._vocab_weight.shape)}", flush=True)

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,
                     draft_step, draft_logits):
        if draft_logits is not None or self.use_local_argmax_reduction or not hasattr(self, "_vocab_ids"):
            return orig_sample(self, hidden_states, sample_src_positions, idx_mapping, temperature,
                               seeds, draft_step, draft_logits)
        logits = torch.nn.functional.linear(hidden_states.to(self._vocab_weight.dtype), self._vocab_weight)
        return self._vocab_ids[logits.argmax(dim=-1)]

    cls.load_model = load_model
    cls.sample_draft = sample_draft


PATCHES = {
    "vllm.v1.worker.gpu.spec_decode.speculator": _patch_speculator,
}
