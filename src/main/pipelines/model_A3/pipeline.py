from kedro.pipeline import Pipeline, node, pipeline
from ..model_common.nodes import (
    make_corpus_text,
    build_tokenizer_spec,
    encode_corpus,
    make_train_val_tensors,
    train_model,
    generate_prompted_samples,
    build_report,
)

def create_pipeline(**kwargs) -> Pipeline:
    mid = "A3"; P = f"params:model_{mid}"
    return pipeline([
        node(make_corpus_text, [f"model_{mid}_raw_corpus", P], f"model_{mid}_corpus_text", name=f"{mid}_corpus_text"),
        node(build_tokenizer_spec, [f"model_{mid}_corpus_text", P], f"model_{mid}_tokenizer_spec", name=f"{mid}_tok_spec"),
        node(encode_corpus, [f"model_{mid}_corpus_text", f"model_{mid}_tokenizer_spec", P], f"model_{mid}_encoded_corpus", name=f"{mid}_encode"),
        node(make_train_val_tensors, [f"model_{mid}_encoded_corpus", P], [f"model_{mid}_train_tensor", f"model_{mid}_val_tensor"], name=f"{mid}_splits"),
        node(train_model, [f"model_{mid}_train_tensor", f"model_{mid}_val_tensor", f"model_{mid}_tokenizer_spec", P], f"model_{mid}_state_dict", name=f"{mid}_train"),
        node(generate_prompted_samples, [f"model_{mid}_state_dict", f"model_{mid}_tokenizer_spec", P], f"model_{mid}_prompted_samples", name=f"{mid}_samples"),
        node(build_report, [f"model_{mid}_state_dict", f"model_{mid}_prompted_samples", P], f"model_{mid}_report", name=f"{mid}_report"),
    ])
