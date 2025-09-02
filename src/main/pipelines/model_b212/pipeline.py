# src/main/pipelines/model_B2/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline
from ..model_common.nodes import (
    build_tokenizer_spec,
    pack_corpus_from_txt_single,   # NEW
    train_model,
    generate_samples_split,
    build_report,
)

def create_pipeline(**kwargs) -> Pipeline:
    mid = "B212"  # keep as-is if this pipeline is for B21
    P = f"params:model_{mid}"
    return pipeline([
        # 1) Tokenizer spec (doesn't really need the text, but signature expects it)
        node(
            func=build_tokenizer_spec,
            inputs=[f"text_only_txt_cased", P],   # pass the combined TXT
            outputs=f"model_{mid}_tokenizer_spec",
            name=f"{mid}_tok_spec",
        ),

        # 2) PACKER: TXT -> (train_tensor, val_tensor) + stats
        node(
            func=pack_corpus_from_txt_single,
            inputs=["text_only_txt_cased", f"model_{mid}_tokenizer_spec", P],
            outputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor", f"model_{mid}_packer_stats"],
            name=f"{mid}_pack_txt",
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
