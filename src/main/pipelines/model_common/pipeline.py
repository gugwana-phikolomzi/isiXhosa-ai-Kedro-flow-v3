# src/main/pipelines/model_common/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline

# import your real node functions here
# from ..model_common.nodes import (
#     make_corpus_text, build_tokenizer_spec, encode_corpus,
#     make_train_val_tensors, train_model, generate_prompted_samples, build_report
# )
# ↓ placeholders: replace with your actual imports
def make_corpus_text(raw_text, params): ...
def build_tokenizer_spec(corpus_text, params): ...
def encode_corpus(corpus_text, tokenizer_spec, params): ...
def make_train_val_tensors(encoded_corpus, params): ...
def train_model(train_tensor, val_tensor, tokenizer_spec, params): ...
def generate_prompted_samples(state_dict, tokenizer_spec, params): ...
def build_report(state_dict, prompted_samples, params): ...

def build_model_pipeline(mid: str) -> Pipeline:
    P = f"params:model_{mid}"
    return pipeline([
        node(make_corpus_text,
             inputs=[f"model_{mid}_raw_corpus", P],
             outputs=f"model_{mid}_corpus_text",
             name=f"{mid}_corpus_text"),
        node(build_tokenizer_spec,
             inputs=[f"model_{mid}_corpus_text", P],
             outputs=f"model_{mid}_tokenizer_spec",
             name=f"{mid}_tok_spec"),
        node(encode_corpus,
             inputs=[f"model_{mid}_corpus_text", f"model_{mid}_tokenizer_spec", P],
             outputs=f"model_{mid}_encoded_corpus",
             name=f"{mid}_encode"),
        node(make_train_val_tensors,
             inputs=[f"model_{mid}_encoded_corpus", P],
             outputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor"],
             name=f"{mid}_splits"),
        node(train_model,
             inputs=[f"model_{mid}_train_tensor", f"model_{mid}_val_tensor", f"model_{mid}_tokenizer_spec", P],
             outputs=f"model_{mid}_state_dict",
             name=f"{mid}_train"),
        node(generate_prompted_samples,
             inputs=[f"model_{mid}_state_dict", f"model_{mid}_tokenizer_spec", P],
             outputs=f"model_{mid}_prompted_samples",
             name=f"{mid}_samples"),
        node(build_report,
             inputs=[f"model_{mid}_state_dict", f"model_{mid}_prompted_samples", P],
             outputs=f"model_{mid}_report",
             name=f"{mid}_report"),
    ])
