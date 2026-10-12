# Put this directory on PYTHONPATH and set SPEC_DRAFT_VOCAB to give the drafter a reduced
# lm_head (spec_vocab.py, vLLM V2 model runner). The patch is applied right after the target
# module is imported, so it also reaches the spawned engine-core / worker processes.
#   SPEC_DRAFT_VOCAB=<top_ids.npy>:<N>   e.g. token_space/top_ids_train.npy:65536
import os

# target module -> patch modules (kept static so torch isn't imported at interpreter start)
_TARGETS: dict[str, list[str]] = {}
if os.environ.get("SPEC_DRAFT_VOCAB"):
    _TARGETS.setdefault("vllm.v1.worker.gpu.spec_decode.speculator", []).append("spec_vocab")

if _TARGETS:
    import importlib
    import importlib.abc
    import sys

    class _PostImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            exec_module = spec.loader.exec_module

            def patched_exec(module):
                exec_module(module)
                for patch_name in _TARGETS[name]:
                    importlib.import_module(patch_name).PATCHES[name](module)

            spec.loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _PostImportHook())
