from kedro.pipeline import Pipeline, node, pipeline
from ..model_common.nodes import (
    make_corpus_text,
    build_tokenizer_spec,
    encode_corpus,
    make_grouped_split_map,      # <-- ensure this is available
    make_train_val_tensors,
    train_model,
    generate_prompted_samples,
    build_report,
)

def create_pipeline(**kwargs) -> Pipeline:
    mid = "B21"; P = f"params:model_{mid}"
    return pipeline([
        node(
            make_corpus_text,
            inputs=[f"model_{mid}_raw_corpus", P],
            outputs=f"model_{mid}_corpus_text",
            name=f"{mid}_corpus_text",
        ),
        node(
            build_tokenizer_spec,
            inputs=[f"model_{mid}_corpus_text", P],
            outputs=f"model_{mid}_tokenizer_spec",
            name=f"{mid}_tok_spec",
        ),
        node(
            encode_corpus,
            inputs=[f"model_{mid}_corpus_text", f"model_{mid}_tokenizer_spec", P],
            outputs=f"model_{mid}_encoded_corpus",
            name=f"{mid}_encode",
        ),
        # Build the grouped, stratified split map BEFORE making tensors
        node(
            make_grouped_split_map,
            inputs=[f"model_{mid}_raw_corpus", P],
            outputs=[f"model_{mid}_split_map", f"model_{mid}_split_stats"],
            name=f"{mid}_splitmap",
        ),
        node(
            make_train_val_tensors,
            inputs=[f"model_{mid}_encoded_corpus", P, f"model_{mid}_split_map"],
            outputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor"],
            name=f"{mid}_splits",
        ),
        node(
            train_model,
            inputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor", f"model_{mid}_tokenizer_spec", P],
            outputs=f"model_{mid}_state_dict",
            name=f"{mid}_train",
        ),
        node(
            generate_prompted_samples,
            inputs=[f"model_{mid}_state_dict", f"model_{mid}_tokenizer_spec", P],
            outputs=f"model_{mid}_prompted_samples",
            name=f"{mid}_samples",
        ),
        node(
            build_report,
            inputs=[f"model_{mid}_state_dict", f"model_{mid}_prompted_samples", P],
            outputs=f"model_{mid}_report",
            name=f"{mid}_report",
        ),
    ])
