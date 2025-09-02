# src/main/pipelines/model_B2/pipeline.py
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

# Nodes from the common module
from ..model_common.nodes import (
    build_tokenizer_spec,
    txt_to_docs_list,          # adapter: text -> [{"id", "text"}...]
    pack_corpus_from_docs,     # length-aware packer (emits train/val + runtime stats)
    train_model,
    generate_samples_split,
    build_report,
    compute_packer_stats,      # NEW: diagnostic stats node (pre-pack)
)

# --- local helper: encode docs -> token-id lists (bytes) for diagnostics ---
def _encode_docs_for_stats(docs: list[dict], tokenizer_spec: dict) -> list[list[int]]:
    """
    Convert [{'text': ...}, ...] to List[List[int]] of UTF-8 byte ids.
    No EOS inserted here; compute_packer_stats will add EOS virtually.
    """
    out: list[list[int]] = []
    for d in docs:
        t = d.get("text") if isinstance(d, dict) else str(d)
        if not t:
            continue
        out.append(list(t.encode("utf-8", errors="ignore")))
    return out


def create_pipeline(**kwargs) -> Pipeline:
    # Keep as-is if this pipeline is used for B21/B213 runs; only the ID string matters for datasets/params.
    mid = "D0"
    P = f"params:model_{mid}"

    return pipeline([
        # 0) Adapter: raw TXT -> all_clean_docs (in-memory)
        node(
            func=txt_to_docs_list,
            inputs=[f"model_{mid}_text_only_txt_cased"],   # data/01_raw/cased.txt
            outputs=f"model_{mid}_all_clean_docs",
            name=f"{mid}_txt_to_docs",
        ),

        # 1) Tokenizer spec (signature expects the text; harmless to pass through)
        node(
            func=build_tokenizer_spec,
            inputs=[f"model_{mid}_text_only_txt_cased", P],
            outputs=f"model_{mid}_tokenizer_spec",
            name=f"{mid}_tok_spec",
        ),

        # 1b) (NEW) Encode docs -> token-id lists for diagnostic stats (pre-pack view)
        node(
            func=_encode_docs_for_stats,
            inputs=[f"model_{mid}_all_clean_docs", f"model_{mid}_tokenizer_spec"],
            outputs=f"model_{mid}_encoded_docs_for_stats",
            name=f"{mid}_encode_docs_for_stats",
        ),

        # 1c) (NEW) Diagnostic packer stats on raw encoded docs (no packing performed here)
        node(
            func=compute_packer_stats,
            inputs=[f"model_{mid}_encoded_docs_for_stats", f"model_{mid}_tokenizer_spec", P],
            outputs=f"model_{mid}_packer_stats_diag",   # <- pre-pack diagnostics JSON
            name=f"{mid}_packer_stats_diag",
        ),

        # 2) PACKER: docs -> train/val tensors + runtime packing stats
        node(
            func=pack_corpus_from_docs,
            inputs=[f"model_{mid}_all_clean_docs", f"model_{mid}_tokenizer_spec", P],
            outputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor", f"model_{mid}_packer_stats"],
            name=f"{mid}_pack_docs",
        ),

        # 3) Train
        node(
            func=train_model,
            inputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor", f"model_{mid}_tokenizer_spec", P],
            outputs=f"model_{mid}_state_dict",
            name=f"{mid}_train",
        ),

        # 4) Samples (random + prompted)
        node(
            func=generate_samples_split,
            inputs=[f"model_{mid}_state_dict", f"model_{mid}_tokenizer_spec", P],
            outputs=[f"model_{mid}_random_samples", f"model_{mid}_prompted_samples"],
            name=f"{mid}_samples",
        ),

        # 5) Report
        node(
            func=build_report,
            inputs=[f"model_{mid}_state_dict", f"model_{mid}_prompted_samples", P],
            outputs=f"model_{mid}_report",
            name=f"{mid}_report",
        ),
    ])
