# src/main/pipelines/model_common/nodes.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
import json, random
from collections import defaultdict, Counter
import numpy as np

# Import your model + helpers
from .gpt import GPT, GPTConfig, configure_adamw, EMA, num_parameters


# ============================================================
# Helpers
# ============================================================

# --- adapter: TXT (one doc per non-empty line) -> list[dict{text}] ---
def txt_to_docs_list(txt: str) -> list[dict]:
    docs = []
    for i, ln in enumerate(txt.splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        docs.append({"id": f"TXT_{i:07d}", "text": ln})
    return docs


def _encode_str_to_byte_ids(s: str) -> List[int]:
    return list(s.encode("utf-8", errors="ignore"))

def _split_into_paragraphs(text: str) -> List[str]:
    """Preserve paragraph boundaries created during cleaning."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if parts:
        return parts
    # fallback: split on single newlines if no blank-line paragraphs
    return [t for t in (seg.strip() for seg in text.split("\n")) if t]

def _greedy_chunk_token_ids(tok_ids: List[int], target: int, overlap: int, min_seg: int) -> List[List[int]]:
    """Slide over a long sequence to produce ~target-length chunks with small overlap."""
    if len(tok_ids) <= target:
        return [tok_ids] if len(tok_ids) >= min_seg else []
    chunks = []
    i = 0
    step = max(target - overlap, 1)
    while i < len(tok_ids):
        j = min(i + target, len(tok_ids))
        seg = tok_ids[i:j]
        if len(seg) >= min_seg:
            chunks.append(seg)
        if j == len(tok_ids):
            break
        i += step
    return chunks


# ============================================================
# 1) Length-aware PACKER (TXT/JSONL -> train/val 1D token tensors)
#     NOTE: Only length-aware chunking/packing. No cleaning here.
# ============================================================

def _docs_to_token_chunks(
    docs: List[Any],
    tokenizer_spec: Dict[str, Any],
    target_chunk_tokens: int,
    split_overlap: int,
    min_segment_tokens: int,
) -> List[List[int]]:
    """
    Convert cleaned docs (strings or dicts w/ 'text') into token chunks near target length.
    No cleaning here; we only segment by paragraphs and tokenize.
    """
    all_chunks: List[List[int]] = []
    for d in docs:
        text = d.get("text") if isinstance(d, dict) else str(d)
        if not text:
            continue
        paras = _split_into_paragraphs(text)
        # tokenize paragraphs separately to allow greedy joins up to target
        para_tok: List[List[int]] = [_encode_str_to_byte_ids(p) for p in paras if p]
        cur: List[int] = []
        for t in para_tok:
            if not t:
                continue
            if len(cur) + len(t) <= target_chunk_tokens:
                cur.extend(t)
            else:
                all_chunks.extend(_greedy_chunk_token_ids(cur, target_chunk_tokens, split_overlap, min_segment_tokens))
                cur = t[:]  # start new chunk with this paragraph
        if cur:
            all_chunks.extend(_greedy_chunk_token_ids(cur, target_chunk_tokens, split_overlap, min_segment_tokens))
    return all_chunks


def _pack_token_chunks(
    chunks: List[List[int]],
    block_size: int,
    min_fill_ratio: float,
    max_fill_ratio: float,
    max_docs_per_pack: int,
    add_eos_between_docs: bool,
    eos_id: Optional[int],
    bucket_edges: List[int],
    seed: int,
) -> Tuple[List[List[int]], Dict[str, float]]:
    """
    Length-aware greedy bin packing of token chunks into sequences <= block_size.
    Buckets by length for speed; shuffles within buckets to avoid order bias.
    """
    random.seed(seed)

    # bucket by length for efficient packing
    buckets: Dict[int, List[List[int]]] = {}
    edges = list(bucket_edges) if bucket_edges else [64, 128, 256, 384, 512, 768, 1024, 10_000]
    for ch in chunks:
        L = len(ch)
        edge = next((e for e in edges if L <= e), edges[-1])
        buckets.setdefault(edge, []).append(ch)

    for b in buckets.values():
        random.shuffle(b)

    min_fill = int(min_fill_ratio * block_size)
    max_fill = int(max_fill_ratio * block_size)

    packs: List[List[int]] = []
    total_len = 0
    shortpacks = 0
    truncated_tokens = 0  # we do not truncate in this packer, but leave for KPI symmetry

    for _, bucket in sorted(buckets.items(), key=lambda kv: kv[0]):  # small->large buckets
        cur, cur_len, cur_docs = [], 0, 0
        for ch in bucket:
            cost = len(ch) + (1 if (add_eos_between_docs and cur_docs > 0) else 0)
            fits_soft = (cur_docs < max_docs_per_pack) and (cur_len + cost <= max_fill)
            if fits_soft:
                if add_eos_between_docs and cur_docs > 0 and eos_id is not None:
                    cur.append(int(eos_id)); cur_len += 1
                cur.extend(ch); cur_len += len(ch); cur_docs += 1
            else:
                # try to reach min_fill once if still within hard block_size
                if (cur_len < min_fill) and (cur_docs < max_docs_per_pack) and (cur_len + cost <= block_size):
                    if add_eos_between_docs and cur_docs > 0 and eos_id is not None:
                        cur.append(int(eos_id)); cur_len += 1
                    cur.extend(ch); cur_len += len(ch); cur_docs += 1
                # flush current
                packs.append(cur); total_len += cur_len
                if cur_len < int(0.3 * block_size):
                    shortpacks += 1
                # start new with ch
                cur, cur_len, cur_docs = [], 0, 0
                cur.extend(ch); cur_len = len(ch); cur_docs = 1
        if cur:
            packs.append(cur); total_len += cur_len
            if cur_len < int(0.3 * block_size):
                shortpacks += 1

    utilization = (total_len / (len(packs) * block_size)) if packs else 0.0
    stats = {
        "num_packs": len(packs),
        "avg_pack_len": (total_len / max(1, len(packs))) if packs else 0,
        "utilization": utilization,
        "shortpack_rate": shortpacks / max(1, len(packs)),
        "truncation_rate": truncated_tokens / max(1, total_len),
        "block_size": block_size,
    }
    return packs, stats


def _split_train_val_packs(packs: List[List[int]], train_frac: float, seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    rnd = random.Random(seed)
    idx = list(range(len(packs)))
    rnd.shuffle(idx)
    cut = int(len(idx) * train_frac)
    train_idx, val_idx = idx[:cut], idx[cut:]
    train = [packs[i] for i in train_idx]
    val = [packs[i] for i in val_idx]
    return train, val


def _concat_packs_to_1d(packs: List[List[int]], eos_id: Optional[int]) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    Flatten packs to a 1D stream and return (stream, ranges) where
    ranges is a list of (start, end) offsets (exclusive) per pack in the flat stream.
    """
    flat: List[int] = []
    ranges: List[Tuple[int,int]] = []
    pos = 0
    for p in packs:
        start = pos
        flat.extend(p); pos += len(p)
        if eos_id is not None:
            flat.append(int(eos_id)); pos += 1
        end = pos  # exclusive
        ranges.append((start, end))
    return flat, ranges


def _build_legal_starts(ranges: List[Tuple[int,int]], block_size: int) -> torch.Tensor:
    """
    Build legal start indices that do not cross pack boundaries when sampling windows of size block_size.
    """
    starts = []
    for s, e in ranges:
        max_start = e - block_size - 1  # need y to have +1
        if max_start >= s:
            starts.extend(range(s, max_start + 1))
    return torch.tensor(starts, dtype=torch.long)


def _pack_docs_to_idstream(docs: List[str], tokenizer_spec: Dict[str, Any], add_eos_between_docs: bool) -> List[int]:
    """Encode each doc, append eos_id between docs if available, and flatten."""
    stream: List[int] = []
    eos_id = tokenizer_spec.get("eos_id") if tokenizer_spec.get("add_special_tokens") else None
    for d in docs:
        stream.extend(_encode_str_to_byte_ids(d))
        if add_eos_between_docs and eos_id is not None:
            stream.append(int(eos_id))
    return stream


def pack_corpus_from_docs(
    docs: List[Any],                 # strings or dicts with 'text'
    tokenizer_spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    NEW entrypoint for modelling: chunk by length, pack to block_size, then split.
    No cleaning; relies on upstream cleaning pipeline to provide docs.
    """
    pcfg = dict(params.get("packer", {}))
    block_size          = int(pcfg.get("block_size", params.get("block_size", 512)))
    train_frac          = float(pcfg.get("train_frac", 0.9))
    seed                = int(pcfg.get("seed", 1337))
    min_fill_ratio      = float(pcfg.get("min_fill_ratio", 0.60))
    max_fill_ratio      = float(pcfg.get("max_fill_ratio", 0.90))
    max_docs_per_pack   = int(pcfg.get("max_docs_per_pack", 3))
    target_chunk_tokens = int(pcfg.get("target_chunk_tokens", int(block_size * 0.75)))
    split_overlap       = int(pcfg.get("split_overlap", 32))
    min_segment_tokens  = int(pcfg.get("min_segment_tokens", 32))
    bucket_edges        = list(pcfg.get("bucket_edges", [64,128,256,384,512,768,1024]))
    add_eos_between_docs= bool(pcfg.get("add_eos_between_docs", True))
    eos_id = tokenizer_spec.get("eos_id") if tokenizer_spec.get("add_special_tokens") else None

    # 1) docs -> token chunks (~target length)
    chunks = _docs_to_token_chunks(
        docs, tokenizer_spec,
        target_chunk_tokens=target_chunk_tokens,
        split_overlap=split_overlap,
        min_segment_tokens=min_segment_tokens,
    )

    # 2) chunks -> packed sequences obeying fill targets
    packs, pack_stats = _pack_token_chunks(
        chunks=chunks,
        block_size=block_size,
        min_fill_ratio=min_fill_ratio,
        max_fill_ratio=max_fill_ratio,
        max_docs_per_pack=max_docs_per_pack,
        add_eos_between_docs=add_eos_between_docs,
        eos_id=eos_id,
        bucket_edges=bucket_edges,
        seed=seed,
    )

    # 3) split AFTER packing to avoid length bias
    train_packs, val_packs = _split_train_val_packs(packs, train_frac=train_frac, seed=seed)

    # 4) concatenate packs to 1D streams for your training sampler (+legal starts)
    train_stream, train_ranges = _concat_packs_to_1d(train_packs, eos_id=eos_id)
    val_stream,   val_ranges   = _concat_packs_to_1d(val_packs,   eos_id=eos_id)

    train_tensor = torch.tensor(train_stream, dtype=torch.long)
    val_tensor   = torch.tensor(val_stream,   dtype=torch.long)

    # optional: record counts of legal starts (not returned to keep signature stable)
    train_starts = _build_legal_starts(train_ranges, block_size)
    val_starts   = _build_legal_starts(val_ranges,   block_size)

    stats = {
        "docs_in": len(docs),
        "chunks_total": len(chunks),
        "packs_total": len(packs),
        "packs_train": len(train_packs),
        "packs_val": len(val_packs),
        "train_tokens": len(train_stream),
        "val_tokens": len(val_stream),
        "train_frac": train_frac,
        "utilization": pack_stats.get("utilization", 0.0),
        "shortpack_rate": pack_stats.get("shortpack_rate", 0.0),
        "truncation_rate": pack_stats.get("truncation_rate", 0.0),
        "block_size": block_size,
        "target_chunk_tokens": target_chunk_tokens,
        "min_fill_ratio": min_fill_ratio,
        "max_fill_ratio": max_fill_ratio,
        "max_docs_per_pack": max_docs_per_pack,
        "split_overlap": split_overlap,
        "min_segment_tokens": min_segment_tokens,
        "bucket_edges": bucket_edges,
        "add_eos_between_docs": add_eos_between_docs,
        "seed": seed,
        "legal_train_starts": int(train_starts.numel()),
        "legal_val_starts": int(val_starts.numel()),
        "note": "Length-aware chunking & pack-after-split for modelling. No cleaning.",
    }
    return train_tensor, val_tensor, stats


def pack_corpus_from_txt_single(
    txt: str,                 # combined TXT; one document per line (tag-free)
    tokenizer_spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Legacy/simple path: docs = lines; split by docs; encode+EOS; concat to 1D streams.
    """
    # --- config
    pcfg = dict(params.get("packer", {}))
    train_frac   = float(pcfg.get("train_frac", 0.9))
    shuffle_docs = bool(pcfg.get("shuffle_docs", True))
    add_eos      = bool(pcfg.get("add_eos_between_docs", True))
    seed         = int(pcfg.get("seed", 1337))

    # --- parse docs
    docs = [ln for ln in txt.splitlines() if ln.strip()]
    n_docs = len(docs)

    # --- shuffle + split by docs (no leakage)
    idx = list(range(n_docs))
    rng = random.Random(seed)
    if shuffle_docs:
        rng.shuffle(idx)
    cut = int(n_docs * train_frac)
    train_docs = [docs[i] for i in idx[:cut]]
    val_docs   = [docs[i] for i in idx[cut:]]

    # --- encode + pack
    train_stream = _pack_docs_to_idstream(train_docs, tokenizer_spec, add_eos)
    val_stream   = _pack_docs_to_idstream(val_docs,   tokenizer_spec, add_eos)

    train_tensor = torch.tensor(train_stream, dtype=torch.long)
    val_tensor   = torch.tensor(val_stream,   dtype=torch.long)

    stats = {
        "docs_total": n_docs,
        "docs_train": len(train_docs),
        "docs_val": len(val_docs),
        "train_tokens": len(train_stream),
        "val_tokens": len(val_stream),
        "train_frac": train_frac,
        "shuffle_docs": shuffle_docs,
        "add_eos_between_docs": add_eos,
        "seed": seed,
        "note": "Split by docs, EOS between docs if tokenizer has eos_id.",
    }
    return train_tensor, val_tensor, stats


def make_corpus_text(raw_text: str, params: Dict[str, Any]) -> str:
    """Simple pass-through; keep as a hook for future cleaning."""
    return raw_text


def build_tokenizer_spec(corpus_text: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Define a byte tokenizer spec with optional special tokens."""
    add_st = bool(params.get("tokenizer", {}).get("add_special_tokens", True))
    vocab_size = 256 + (2 if add_st else 0)
    spec = {
        "type": "byte",
        "add_special_tokens": add_st,
        "vocab_size": vocab_size,
        "bos_id": 256 if add_st else None,
        "eos_id": 257 if add_st else None,
    }
    return spec


def encode_corpus(corpus_text: str, tokenizer_spec: Dict[str, Any], params: Dict[str, Any]) -> torch.Tensor:
    """Encode text -> list of byte IDs; optionally wrap with BOS/EOS."""
    ids = _encode_str_to_byte_ids(corpus_text)
    if tokenizer_spec.get("add_special_tokens"):
        bos = tokenizer_spec["bos_id"]
        eos = tokenizer_spec["eos_id"]
        ids = [bos] + ids + [eos]
    return torch.tensor(ids, dtype=torch.long)


def make_train_val_tensors(
    encoded_corpus,
    params: Dict[str, Any],
    split_map: Dict[str, str] | None = None,   # optional: {doc_id: "train"/"val"}
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward-compatible splitter.

    If 'encoded_corpus' is a plain 1D tensor (B2-style), do the original
    contiguous split by 'train_frac'.

    If 'encoded_corpus' is the dict produced by encode_corpus(corpus_format="jsonl"),
    and 'split_map' is provided, build train/val by concatenating full documents,
    inserting EOS *between docs only*. This prevents leakage and respects groups.
    """
    # Legacy path: single tensor -> fractional split
    if isinstance(encoded_corpus, torch.Tensor):
        frac = float(params.get("train_frac", 0.9))
        n = encoded_corpus.numel()
        n_train = int(n * frac)
        train = encoded_corpus[:n_train].contiguous()
        val = encoded_corpus[n_train:].contiguous()
        return train, val

    # Dict path (JSONL-aware)
    assert isinstance(encoded_corpus, dict), "encoded_corpus must be Tensor or dict"
    ids_tensor: torch.Tensor = encoded_corpus["ids"]
    docs: List[List[int]] = encoded_corpus.get("docs", [])
    doc_ids: List[str] = encoded_corpus.get("doc_ids", [])
    add_st: bool = bool(encoded_corpus.get("add_special_tokens", True))

    # If no split_map was provided, fall back to fractional split on flat tensor
    if not split_map:
        frac = float(params.get("train_frac", 0.9))
        n = ids_tensor.numel()
        n_train = int(n * frac)
        train = ids_tensor[:n_train].contiguous()
        val = ids_tensor[n_train:].contiguous()
        return train, val

    # Build train/val streams by whole docs
    eos_id = None
    if add_st:
        # build_tokenizer_spec sets eos_id = 257 when add_special_tokens=True
        eos_id = 257

    strip_tag_header = bool(params.get("eval_strip_first_line", False))

    train_flat: List[int] = []
    val_flat: List[int] = []

    for di, did in zip(docs, doc_ids):
        split = str(split_map.get(did, "train")).lower()  # default to train if missing
        # Optionally strip first line (header tag) for *validation* only
        di_use = di
        if strip_tag_header and split == "val":
            try:
                nl = di.index(10)  # byte value for '\n'
                di_use = di[nl + 1 :]
            except ValueError:
                di_use = di  # no newline -> leave as-is

        if split == "val":
            val_flat.extend(di_use)
            if eos_id is not None:
                val_flat.append(eos_id)
        else:
            train_flat.extend(di_use)
            if eos_id is not None:
                train_flat.append(eos_id)

    train_tensor = torch.tensor(train_flat, dtype=torch.long)
    val_tensor = torch.tensor(val_flat, dtype=torch.long)
    return train_tensor, val_tensor


def make_grouped_split_map(
    raw_jsonl: str,   # the combined JSONL string
    params: dict
) -> tuple[dict, dict]:
    """
    Build a grouped, stratified train/val split map for B21+.

    Returns:
      - split_map: {doc_id: "train"/"val"}
      - stats: split stats for reporting
    """
    train_frac = float(params.get("train_frac", 0.9))
    seed = int(params.get("seed", 1337))
    random.seed(seed)

    lines = [l for l in raw_jsonl.splitlines() if l.strip()]
    docs = [json.loads(l) for l in lines]

    # group keys: Bible → section; News → url (fallback id)
    grouped = defaultdict(list)
    for d in docs:
        domain = d.get("domain")
        if domain == "BIBLE":
            key = d.get("section")
        else:
            key = d.get("url") or d.get("id")
        grouped[(domain, key)].append(d)

    split_map = {}
    stats = {"train": {"NEWS": 0, "BIBLE": 0}, "val": {"NEWS": 0, "BIBLE": 0}}

    for domain in ["NEWS", "BIBLE"]:
        groups = [g for g in grouped if g[0] == domain]
        random.shuffle(groups)
        cut = int(len(groups) * train_frac)
        train_g, val_g = groups[:cut], groups[cut:]
        for g in train_g:
            for d in grouped[g]:
                split_map[d["id"]] = "train"
                stats["train"][domain] += 1
        for g in val_g:
            for d in grouped[g]:
                split_map[d["id"]] = "val"
                stats["val"][domain] += 1

    stats["total"] = {"train": sum(stats["train"].values()), "val": sum(stats["val"].values())}
    return split_map, stats


# ============================================================
# 2) Training / evaluation utilities
# ============================================================

def _get_device(params: Dict[str, Any]) -> torch.device:
    if params.get("device_auto_cuda", False) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_torch_threads(params: Dict[str, Any]) -> None:
    th = int(params.get("torch_threads", max(1, os.cpu_count() or 1)))
    torch.set_num_threads(th)


def _lr_getter(params: Dict[str, Any], max_steps: int):
    """Return a function f(step)->lr implementing warmup + chosen schedule."""
    sched = str(params.get("lr_schedule", "cosine")).lower()
    warmup = int(params.get("warmup_steps", 0))
    base_lr = float(params.get("learning_rate", 6e-4))
    min_lr = float(params.get("min_lr", 0.0))

    def lr(step: int) -> float:
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        t = step - warmup
        T = max(1, max_steps - warmup)
        if sched == "linear":
            pct = max(0.0, 1.0 - t / T)
            return min_lr + (base_lr - min_lr) * pct
        elif sched == "cosine":
            cos = 0.5 * (1 + math.cos(math.pi * t / T))
            return min_lr + (base_lr - min_lr) * cos
        # constant fallback
        return base_lr

    return lr


def _get_batch(split_tensor: torch.Tensor, block_size: int, batch_size: int, device: torch.device,
               legal_starts: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (x, y) from a 1D token tensor; optionally restrict to legal pack starts."""
    if legal_starts is None:
        n = split_tensor.numel() - block_size - 1
        idx = torch.randint(0, max(1, n), (batch_size,), device=device)
    else:
        if legal_starts.device != device:
            legal_starts = legal_starts.to(device)
        choice_idx = torch.randint(0, legal_starts.numel(), (batch_size,), device=device)
        idx = legal_starts[choice_idx]
    x = torch.stack([split_tensor[i:i + block_size] for i in idx]).to(device)
    y = torch.stack([split_tensor[i + 1:i + 1 + block_size] for i in idx]).to(device)
    return x, y


@torch.no_grad()
def _estimate_loss(model: nn.Module, train: torch.Tensor, val: torch.Tensor, block_size: int, batch_size: int, iters: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    out = {}
    for name, split in [("train", train), ("val", val)]:
        losses = []
        for _ in range(max(1, iters)):
            xb, yb = _get_batch(split, block_size, batch_size, device)
            logits, loss = model(xb, yb)
            losses.append(loss.item())
        out[f"{name}_loss"] = float(sum(losses) / len(losses))
    model.train()
    return out


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ============================================================
# 3) Train node
# ============================================================

def train_model(
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    tokenizer_spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train GPT on CPU (or CUDA if allowed) with early stopping.
    Returns a dict containing the **BEST** model state (EMA if available and preferred), plus metadata.
    """
    _set_torch_threads(params)
    device = _get_device(params)

    # --- Build model config
    cfg = GPTConfig(
        vocab_size=int(tokenizer_spec["vocab_size"]),
        n_layer=int(params["n_layer"]),
        n_head=int(params["n_head"]),
        n_embd=int(params["n_embd"]),
        block_size=int(params["block_size"]),
        dropout=float(params.get("dropout", 0.0)),
        bias=True,
    )
    model = GPT(cfg).to(device)

    # --- Optimizer (AdamW with hygiene if provided)
    opt = configure_adamw(
        model,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        beta2=float(params.get("adam_beta2", 0.999)),
        eps=float(params.get("adam_eps", 1e-8)),
        no_decay_patterns=params.get("no_decay_patterns"),
    )

    # --- EMA (optional)
    ema_cfg = params.get("ema", {})
    use_ema = bool(ema_cfg.get("enabled", False))
    ema = EMA(model, decay=float(ema_cfg.get("decay", 0.999)), start_step=int(ema_cfg.get("start_step", 1000))) if use_ema else None
    prefer_ema_for_sampling = bool(ema_cfg.get("eval_with_ema", True))  # default True

    # --- Training hyperparams
    seed = int(params.get("seed", 1337))
    torch.manual_seed(seed)
    batch_size = int(params["batch_size"])
    grad_accum = int(params.get("grad_accum_steps", 1))
    block_size = int(params["block_size"])
    max_iters = int(params.get("target_total_steps") or params.get("max_iters", 20000))
    eval_interval = int(params.get("eval_interval", 250))
    eval_iters = int(params.get("eval_iters", 40))
    clip = float(params.get("grad_clip_norm", 1.0))

    # Early stopping
    es = params.get("early_stopping", {})
    es_enabled = bool(es.get("enabled", True))
    patience = int(es.get("patience_evals", 5))
    min_improve_pct = float(es.get("min_improve_pct", 1.0)) / 100.0
    min_delta_abs = float(es.get("min_delta_abs", 0.0))

    # Checkpointing
    ckpt_dir = params.get("checkpoint_dir", f"data/06_models/{params.get('experiment_id','model')}/checkpoints")
    ckpt_prefix = params.get("checkpoint_prefix", params.get("experiment_id", "model"))
    ckpt_every = int(params.get("checkpoint_interval", 500))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics_jsonl = params.get("metrics_jsonl_path", f"data/08_reporting/{params.get('experiment_id','model')}/metrics.jsonl")
    metrics_csv   = params.get("metrics_csv_path",   f"data/08_reporting/{params.get('experiment_id','model')}/metrics.csv")
    Path(os.path.dirname(metrics_jsonl)).mkdir(parents=True, exist_ok=True)

    # LR schedule
    lr_of = _lr_getter(params, max_iters)

    # Throughput helpers
    tok_per_step  = batch_size * grad_accum * block_size
    param_count   = num_parameters(model, trainable_only=False)

    # Best tracking (in-memory snapshots)
    best_val = float("inf")
    best_step = 0
    best_model_state: Dict[str, torch.Tensor] | None = None
    best_ema_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    t0 = time.time()

    # CSV header if new
    if not os.path.exists(metrics_csv):
        _ensure_dir(metrics_csv)
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("experiment_id,step,train_loss,val_loss,ppl_v,lr,gn,tok_s\n")

    for step in range(1, max_iters + 1):
        # set LR
        for g in opt.param_groups:
            g["lr"] = lr_of(step - 1)

        t_step = time.time()
        total_loss = 0.0

        # grad accumulation
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            xb, yb = _get_batch(train_tensor, block_size, batch_size, device)
            _, loss = model(xb, yb)
            loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item()

        # clip + step
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item())
        opt.step()
        if use_ema:
            ema.update(model, step)
        step_time = time.time() - t_step
        tok_s = int(tok_per_step / max(step_time, 1e-9))

        # periodic eval
        if (step % eval_interval == 0) or (step == 1):
            eval_model = ema.eval_model() if (use_ema and prefer_ema_for_sampling) else model
            losses = _estimate_loss(eval_model, train_tensor, val_tensor, block_size, batch_size, eval_iters, device)
            tr_loss = float(losses["train_loss"])
            va_loss = float(losses["val_loss"])
            ppl_v = math.exp(va_loss)

            print(f"[{params.get('experiment_id','model')}] step {step} "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} ppl_v={ppl_v:.2f} "
                  f"lr={opt.param_groups[0]['lr']:.6f} gn={gn:.3f} tok/s={tok_s}")

            # metrics sinks
            row = {
                "experiment_id": params.get("experiment_id", "model"),
                "step": step,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "ppl": ppl_v,
                "lr": opt.param_groups[0]["lr"],
                "gn": gn,
                "tok_s": tok_s,
                "params": param_count,
            }
            _append_jsonl(metrics_jsonl, row)
            with open(metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{row['experiment_id']},{step},{tr_loss:.6f},{va_loss:.6f},{ppl_v:.6f},{row['lr']:.8f},{gn:.6f},{tok_s}\n")

            # early-stopping bookkeeping
            improved_rel = (best_val - va_loss) > (min_improve_pct * max(best_val, 1e-8))
            improved_abs = (best_val - va_loss) > min_delta_abs
            improved = (best_val == float("inf")) or improved_rel or improved_abs
            if improved:
                best_val = va_loss
                best_step = step

                # snapshot the current **raw** model weights
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # snapshot EMA weights as well (if enabled)
                if use_ema:
                    best_ema_state = {k: v.detach().cpu().clone() for k, v in ema.eval_model().state_dict().items()}

                # save "best" checkpoint to disk
                out_best = os.path.join(ckpt_dir, f"{ckpt_prefix}_step_{step}_best.pt")
                torch.save({
                    "model": best_model_state,
                    "ema": best_ema_state,
                    "config": asdict(cfg),
                    "tokenizer_spec": tokenizer_spec,
                    "step": step,
                    "best_val": best_val,
                    "params": param_count,
                }, out_best)

                # reset no-improve counter
                no_improve = 0
            else:
                no_improve += 1

            # periodic checkpoint (non-best)
            if ckpt_every and (step % ckpt_every == 0):
                out = os.path.join(ckpt_dir, f"{ckpt_prefix}_step_{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "tokenizer_spec": tokenizer_spec,
                    "step": step,
                    "best_val": best_val
                }, out)

            # stop?
            if es_enabled and (no_improve >= patience):
                print(f"[{params.get('experiment_id','model')}] Early stopping at step {step} "
                      f"(best_val={best_val:.6f} @ {best_step}, no_improve_evals={no_improve}).")
                break

    wall = time.time() - t0

    # ---- decide what to RETURN for downstream sampling ----
    # Prefer EMA best for inference if available & preferred; else raw best.
    chosen_state = None
    if use_ema and prefer_ema_for_sampling and best_ema_state is not None:
        chosen_state = best_ema_state
    elif best_model_state is not None:
        chosen_state = best_model_state
    else:
        # fallback if no improvement was recorded (edge case)
        chosen_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    result = {
        "model_state_dict": chosen_state,              # <-- BEST (EMA if preferred)
        "ema_state_dict": best_ema_state,              # keep for reference
        "config": asdict(cfg),
        "tokenizer_spec": tokenizer_spec,
        "best_val": best_val,
        "best_step": best_step,
        "params": param_count,
        "wallclock_seconds": wall,
    }
    return result


# ============================================================
# 4) Sampling / report nodes
# ============================================================

@torch.no_grad()
def _sample_tokens(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Token sampler with greedy fallback.
      - Greedy if (temperature <= 0) or (top_p >= 1.0 and top_k <= 0)
      - Otherwise: temperature + (top-k/top-p) stochastic sampling
    """
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    greedy = (temperature is None or temperature <= 0) and (top_p >= 1.0) and (top_k <= 0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond, None)
        logits = logits[:, -1, :]  # (B, vocab)

        if greedy:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # temperature
            if temperature and temperature > 0:
                logits = logits / float(temperature)

            # top-k
            if top_k and top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                logits = mask.scatter(-1, topk_idx, topk_vals)

            # top-p (nucleus)
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                # mask tokens where cumulative prob > top_p (keep at least 1)
                cutoff = cumprobs > top_p
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float("-inf")
                # unsort back to original index order
                unsorted = torch.full_like(logits, float("-inf"))
                logits = unsorted.scatter(-1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    return idx


def generate_prompted_samples(
    state_bundle: dict,
    tokenizer_spec: dict,
    params: dict,
) -> dict:
    cfg = GPTConfig(**state_bundle["config"])
    device = _get_device(params)
    model = GPT(cfg).to(device)

    # prefer EMA if present
    sd = state_bundle.get("model_state_dict") or {}
    ema_sd = state_bundle.get("ema_state_dict") or {}
    model.load_state_dict(ema_sd or sd, strict=True)
    model.eval()

    bos_id = tokenizer_spec.get("bos_id")
    eos_id = tokenizer_spec.get("eos_id") if tokenizer_spec.get("add_special_tokens") else None

    # defaults / overrides
    max_tokens = int(params.get("sample_tokens", 150))
    top_p      = float(params.get("sample_top_p", 0.90))
    top_k      = int(params.get("sample_top_k", 50))
    temperature_random = float(params.get("sample_random_temperature", 0.8))
    temperature_prompt = float(params.get("sample_prompt_temperature", 0.7))

    random_pairs = params.get("sample_random_pairs") or [(0.7,111), (0.9,222), (0.7,333), (0.9,444)]
    prompts = params.get("sample_prompts") or [
        "Bhala isiqendu esifutshane sichaza inkqubo yokuvota eMzantsi Afrika.",
        "Qalisa ibali: 'Kwizolo kusasa, ndiphume ndisiya eTaxi Rank...'",
        "Phendula nge-JSON enezitshixo `topic`, `bullets` (3) ngomxholo: 'Ukulungiselela udliwanondlebe lomsebenzi'.",
        "Nika uluhlu lwezixeko ezi-5 eMpuma Koloni ngesiXhosa, uze uchaze esinye ngesiNgesi kwisivakalisi esinye.",
    ]

    def _encode_prompt(p: str) -> torch.Tensor:
        ids = _encode_str_to_byte_ids(p)
        if tokenizer_spec.get("add_special_tokens") and bos_id is not None:
            ids = [bos_id] + ids
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def _decode_bytes(seq: list[int]) -> str:
        if tokenizer_spec.get("add_special_tokens"):
            seq = [t for t in seq if t < 256]
        return bytes(seq).decode("utf-8", errors="ignore")

    # ---- 4 RANDOM (empty prompt) ----
    random_out = []
    empty = ""
    for t, s in random_pairs:
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        x = _encode_prompt(empty)
        y = _sample_tokens(
            model, x, max_new_tokens=max_tokens, eos_id=eos_id,
            temperature=float(t if t is not None else temperature_random),
            top_p=top_p, top_k=top_k,
        )
        text = _decode_bytes(y[0].tolist())
        random_out.append({
            "kind": "random",
            "prompt": empty,
            "temperature": float(t),
            "seed": int(s),
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_tokens,
            "text": text,
        })

    # ---- 4 PROMPTED ----
    prompted_out = []
    T = temperature_prompt
    S = int(params.get("sample_prompt_seed", 333))
    for pr in prompts:
        random.seed(S); np.random.seed(S); torch.manual_seed(S)
        x = _encode_prompt(pr)
        y = _sample_tokens(
            model, x, max_new_tokens=max_tokens, eos_id=eos_id,
            temperature=T, top_p=top_p, top_k=top_k,
        )
        text = _decode_bytes(y[0].tolist())
        prompted_out.append({
            "kind": "prompted",
            "prompt": pr,
            "temperature": T,
            "seed": S,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_tokens,
            "text": text,
        })

    # Optional: legacy mapping
    by_prompt = {row["prompt"]: row["text"] for row in prompted_out}

    return {
        "random": random_out,
        "prompted": prompted_out,
        "by_prompt": by_prompt,
        "meta": {
            "best_step": state_bundle.get("best_step"),
            "best_val": state_bundle.get("best_val"),
            "used_ema": bool(ema_sd),
        },
    }

def generate_samples_split(state_bundle, tokenizer_spec, params):
    combined = generate_prompted_samples(state_bundle, tokenizer_spec, params)
    return combined["random"], combined["prompted"]


def build_report(
    state_bundle: Dict[str, Any],
    prompted_samples: Dict[str, str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Small JSON report with best metrics and a few generations."""
    report = {
        "experiment_id": params.get("experiment_id", "model"),
        "best_val": float(state_bundle.get("best_val", float("nan"))),
        "best_step": int(state_bundle.get("best_step", -1)),
        "params": int(state_bundle.get("params", 0)),
        "wallclock_seconds": float(state_bundle.get("wallclock_seconds", 0.0)),
        "samples": prompted_samples,
        "config": state_bundle.get("config"),
    }
    return report


# ============================================================
# 5) Packer STATS node (diagnostic only; does not alter data)
# ============================================================

def _infer_eos_id(tokenizer_spec: Dict[str, Any], fallback_eos_id: int | None = None) -> int:
    """
    Try to discover the EOS id from tokenizer_spec. Supports a few common shapes:
      - tokenizer_spec['eos_id']
      - tokenizer_spec['special_tokens']['<EOS>'] or ['eos']
      - tokenizer_spec['stoi']['<EOS>']
    If not found, uses fallback_eos_id; if still None, raises.
    """
    if tokenizer_spec is None:
        tokenizer_spec = {}
    # direct common field for our byte tokenizer
    if "eos_id" in tokenizer_spec and isinstance(tokenizer_spec["eos_id"], int):
        return int(tokenizer_spec["eos_id"])

    candidates = []
    for key in ("special_tokens", "stoi", "vocab", "tokens"):
        val = tokenizer_spec.get(key)
        if isinstance(val, dict):
            candidates.append(val)

    for d in candidates:
        for k in ("<EOS>", "<eos>", "eos", "EOS"):
            if k in d and isinstance(d[k], int):
                return int(d[k])

    if fallback_eos_id is not None:
        return int(fallback_eos_id)

    raise ValueError("Could not infer eos_id from tokenizer_spec; please provide params.eos_id.")


def _flatten_with_doc_ids(
    docs_tokens: List[List[int]], eos_id: int
) -> Tuple[List[int], List[int]]:
    """
    Concatenate docs with an EOS token after each doc.
    Returns:
      tokens: flat token stream
      doc_ids: parallel stream of doc indices (same length as tokens)
    """
    flat_tokens: List[int] = []
    flat_doc_ids: List[int] = []
    for di, toks in enumerate(docs_tokens):
        if not toks:
            flat_tokens.append(eos_id)
            flat_doc_ids.append(di)
            continue
        flat_tokens.extend(toks)
        flat_doc_ids.extend([di] * len(toks))
        flat_tokens.append(eos_id)
        flat_doc_ids.append(di)
    return flat_tokens, flat_doc_ids


def _slice_chunks_for_stats(
    xs: List[int], ys: List[int], block_size: int
) -> List[Tuple[List[int], List[int]]]:
    """
    Drop the tail remainder that doesn't fit exactly (typical LM training behavior).
    Returns list of (tokens_slice, doc_ids_slice) pairs of length block_size each.
    """
    assert len(xs) == len(ys)
    n_full = len(xs) // block_size
    out = []
    for i in range(n_full):
        s = i * block_size
        e = s + block_size
        out.append((xs[s:e], ys[s:e]))
    return out


def _chunk_doc_mix_stats(doc_ids_chunk: List[int]) -> Dict[str, Any]:
    """Compute per-chunk mix stats: switches and unique_docs."""
    switches = sum(1 for i in range(1, len(doc_ids_chunk)) if doc_ids_chunk[i] != doc_ids_chunk[i - 1])
    unique_docs = len(set(doc_ids_chunk))
    return {"switches": switches, "unique_docs": unique_docs}


def _histogram(values: List[int], bins: List[int]) -> Dict[str, int]:
    """
    Bucket values into integer bins defined by right-closed edges.
    Example bins: [64, 128, 192, 256, 384, 512, 768, 1024]
    Returns dict like {"<= 64": 10, ..., "> 1024": 3}
    """
    counts = Counter()
    for v in values:
        placed = False
        for b in bins:
            if v <= b:
                counts[f"<= {b}"] += 1
                placed = True
                break
        if not placed:
            counts[f"> {bins[-1]}"] += 1
    ordered = {}
    keys = [f"<= {b}" for b in bins] + [f"> {bins[-1]}"]
    for k in keys:
        if k in counts:
            ordered[k] = counts[k]
    return ordered


def compute_packer_stats(
    encoded_corpus: List[List[int]],
    tokenizer_spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute comprehensive packing stats for a list of tokenized documents.

    Inputs
    ------
    encoded_corpus: List[List[int]]  (token ids per document; no literal <EOS> inside)
    tokenizer_spec: Dict             (used to infer eos_id)
    params: Dict                     (expects 'block_size', optional 'eos_id', optional 'doc_len_hist_bins')

    Output
    ------
    stats: Dict with sections: corpus, doc_len, packing, alignment
    """
    block_size = int(params["block_size"])
    eos_id = _infer_eos_id(tokenizer_spec, params.get("eos_id"))
    bins = params.get("doc_len_hist_bins", [64, 128, 192, 256, 384, 512, 768, 1024])

    # --- doc length stats (pre-EOS) ---
    doc_lens = [len(d) for d in encoded_corpus]
    if not doc_lens:
        raise ValueError("encoded_corpus is empty; cannot compute packer stats.")

    token_count_no_eos = int(sum(doc_lens))
    # prepare flat stream with EOS between docs
    flat_tokens, flat_doc_ids = _flatten_with_doc_ids(encoded_corpus, eos_id)
    token_count_with_eos = len(flat_tokens)
    eos_count = flat_tokens.count(eos_id)
    eos_rate = eos_count / max(1, token_count_with_eos)

    # chunking
    chunks = _slice_chunks_for_stats(flat_tokens, flat_doc_ids, block_size)
    chunk_count = len(chunks)
    tail_waste_tokens = token_count_with_eos - (chunk_count * block_size)
    tail_waste_rate = tail_waste_tokens / max(1, token_count_with_eos)

    # per-chunk mix stats
    switches_list = []
    unique_docs_list = []
    unique_docs_dist = Counter()
    switches_dist = Counter()
    for toks, dids in chunks:
        st = _chunk_doc_mix_stats(dids)
        switches_list.append(st["switches"])
        unique_docs_list.append(st["unique_docs"])
        unique_key = st["unique_docs"] if st["unique_docs"] <= 3 else ">=4"
        switches_key = st["switches"] if st["switches"] <= 3 else ">=4"
        unique_docs_dist[str(unique_key)] += 1
        switches_dist[str(switches_key)] += 1

    # alignment buckets vs context length
    ctx = block_size
    le25 = sum(1 for L in doc_lens if L <= 0.25 * ctx)
    b25_50 = sum(1 for L in doc_lens if 0.25 * ctx < L <= 0.50 * ctx)
    b50_75 = sum(1 for L in doc_lens if 0.50 * ctx < L <= 0.75 * ctx)
    b75_100 = sum(1 for L in doc_lens if 0.75 * ctx < L <= ctx)
    gt = sum(1 for L in doc_lens if L > ctx)
    n_docs = len(doc_lens)

    # basic stats
    doc_len_np = np.array(doc_lens, dtype=np.int64)
    doc_len_stats = {
        "max": int(doc_len_np.max()),
        "min": int(doc_len_np.min()),
        "mean": float(doc_len_np.mean()),
        "median": float(np.median(doc_len_np)),
        "pctiles": {
            "p50": float(np.percentile(doc_len_np, 50)),
            "p75": float(np.percentile(doc_len_np, 75)),
            "p90": float(np.percentile(doc_len_np, 90)),
            "p95": float(np.percentile(doc_len_np, 95)),
            "p99": float(np.percentile(doc_len_np, 99)),
        },
        "hist": _histogram(doc_lens, bins),
    }

    stats = {
        "corpus": {
            "doc_count": n_docs,
            "token_count_no_eos": token_count_no_eos,
            "token_count_with_eos": token_count_with_eos,
            "eos_id": eos_id,
            "eos_count": eos_count,
            "eos_rate": eos_rate,
        },
        "doc_len": doc_len_stats,
        "packing": {
            "block_size": block_size,
            "chunk_count": chunk_count,
            "tail_waste_tokens": tail_waste_tokens,
            "tail_waste_rate": tail_waste_rate,
            "avg_switches_per_chunk": float(np.mean(switches_list)) if switches_list else 0.0,
            "avg_unique_docs_per_chunk": float(np.mean(unique_docs_list)) if unique_docs_list else 0.0,
            "chunks_by_unique_docs": {
                "1": unique_docs_dist.get("1", 0),
                "2": unique_docs_dist.get("2", 0),
                "3": unique_docs_dist.get("3", 0),
                ">=4": unique_docs_dist.get(">=4", 0),
            },
            "switches_hist": {
                "0": switches_dist.get("0", 0),
                "1": switches_dist.get("1", 0),
                "2": switches_dist.get("2", 0),
                "3": switches_dist.get("3", 0),
                ">=4": switches_dist.get(">=4", 0),
            },
        },
        "alignment": {
            "pct_docs_<=25pct_ctx": le25 / n_docs,
            "pct_docs_25_50pct_ctx": b25_50 / n_docs,
            "pct_docs_50_75pct_ctx": b50_75 / n_docs,
            "pct_docs_75_100pct_ctx": b75_100 / n_docs,
            "pct_docs_>ctx": gt / n_docs,
        },
    }
    return stats
