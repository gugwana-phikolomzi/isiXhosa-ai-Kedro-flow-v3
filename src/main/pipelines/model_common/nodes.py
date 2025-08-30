# src/main/pipelines/model_common/nodes.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
import torch.nn.functional as F
from torch import nn
import json, random
from collections import defaultdict

# Import your model + helpers
from .gpt import GPT, GPTConfig, configure_adamw, EMA, num_parameters


# ---------- 1) Data prep nodes ----------

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


def _encode_str_to_byte_ids(s: str) -> List[int]:
    return list(s.encode("utf-8", errors="ignore"))


def encode_corpus(corpus_text: str, tokenizer_spec: Dict[str, Any], params: Dict[str, Any]) -> torch.Tensor:
    """Encode text -> list of byte IDs; optionally wrap with BOS/EOS."""
    ids = _encode_str_to_byte_ids(corpus_text)
    if tokenizer_spec.get("add_special_tokens"):
        bos = tokenizer_spec["bos_id"]
        eos = tokenizer_spec["eos_id"]
        ids = [bos] + ids + [eos]
    return torch.tensor(ids, dtype=torch.long)


# def make_train_val_tensors(encoded_corpus: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Split a single long tensor of token IDs into train/val 1D tensors."""
#     frac = float(params.get("train_frac", 0.9))
#     n = encoded_corpus.numel()
#     n_train = int(n * frac)
#     train = encoded_corpus[:n_train].contiguous()
#     val = encoded_corpus[n_train:].contiguous()
#     return train, val
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

    Optional eval nicety:
      If params['eval_strip_first_line'] is True, we remove the first line
      (up to the first newline byte) from each *val* document before concatenation.
      This is useful when your docs begin with a tag header like "<NEWS>\n".
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
    # Special tokens: we will *not* add BOS per doc; we only insert EOS between docs.
    eos_id = None
    if add_st:
        # Recreate eos_id from tokenizer spec assumptions:
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
            # remove bytes up to and including the first '\n' (0x0A)
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
    Build a grouped, stratified train/val split map for B21.

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


# ---------- 2) Training / evaluation utilities ----------

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


def _get_batch(split_tensor: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (x, y) from a 1D token tensor."""
    n = split_tensor.numel() - block_size - 1
    idx = torch.randint(0, max(1, n), (batch_size,), device=device)
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


# ---------- 3) Train node ----------

def train_model(
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    tokenizer_spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train GPT on CPU (or CUDA if allowed) with early stopping.
    Returns a dict containing model state and metadata (pickled by Kedro).
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
    metrics_csv = params.get("metrics_csv_path",  f"data/08_reporting/{params.get('experiment_id','model')}/metrics.csv")
    Path(os.path.dirname(metrics_jsonl)).mkdir(parents=True, exist_ok=True)

    # LR schedule
    lr_of = _lr_getter(params, max_iters)

    # Throughput helpers
    tok_per_step = batch_size * grad_accum * block_size
    param_count = num_parameters(model, trainable_only=False)

    # Training loop
    best_val = float("inf")
    best_step = 0
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
        for micro in range(grad_accum):
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
            eval_model = ema.eval_model() if (use_ema and ema_cfg.get("eval_with_ema", True)) else model
            losses = _estimate_loss(eval_model, train_tensor, val_tensor, block_size, batch_size, eval_iters, device)
            tr_loss = float(losses["train_loss"])
            va_loss = float(losses["val_loss"])
            ppl_v = math.exp(va_loss)

            # log line
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
                no_improve = 0
                # save "best" checkpoint
                out_best = os.path.join(ckpt_dir, f"{ckpt_prefix}_step_{step}_best.pt")
                torch.save({"model": model.state_dict(),
                            "ema": (ema.eval_model().state_dict() if use_ema else None),
                            "config": asdict(cfg),
                            "tokenizer_spec": tokenizer_spec,
                            "step": step,
                            "best_val": best_val,
                            "params": param_count}, out_best)
            else:
                no_improve += 1

            # periodic checkpoint (non-best)
            if ckpt_every and (step % ckpt_every == 0):
                out = os.path.join(ckpt_dir, f"{ckpt_prefix}_step_{step}.pt")
                torch.save({"model": model.state_dict(),
                            "config": asdict(cfg),
                            "tokenizer_spec": tokenizer_spec,
                            "step": step,
                            "best_val": best_val}, out)

            # stop?
            if es_enabled and (no_improve >= patience):
                print(f"[{params.get('experiment_id','model')}] Early stopping at step {step} "
                      f"(best_val={best_val:.6f} @ {best_step}, no_improve_evals={no_improve}).")
                break

    wall = time.time() - t0

    # final export (return something picklable)
    result = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": (ema.eval_model().state_dict() if use_ema else None),
        "config": asdict(cfg),
        "tokenizer_spec": tokenizer_spec,
        "best_val": best_val,
        "best_step": best_step,
        "params": param_count,
        "wallclock_seconds": wall,
    }
    return result


# ---------- 4) Sampling / report nodes ----------

@torch.no_grad()
def _sample_tokens(
    model: GPT, idx: torch.Tensor, max_new_tokens: int, eos_id: int | None
) -> torch.Tensor:
    """Greedy sampling; stops on EOS if provided."""
    model.eval()
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond, None)
        logits = logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and (next_id.item() == eos_id):
            break
    return idx


def generate_prompted_samples(
    state_bundle: Dict[str, Any],
    tokenizer_spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate continuations from fixed prompts; return dict prompt->text."""
    cfg = GPTConfig(**state_bundle["config"])
    device = _get_device(params)
    model = GPT(cfg).to(device)
    model.load_state_dict(state_bundle["model_state_dict"])
    model.eval()

    bos_id = tokenizer_spec.get("bos_id")
    eos_id = tokenizer_spec.get("eos_id") if tokenizer_spec.get("add_special_tokens") else None

    prompts = params.get("sample_prompts", ["Molo ", "Ulwimi lwesiXhosa "])
    max_tokens = int(params.get("sample_tokens", 150))
    out: Dict[str, str] = {}

    for p in prompts:
        # encode prompt
        ids = _encode_str_to_byte_ids(p)
        if tokenizer_spec.get("add_special_tokens") and bos_id is not None:
            ids = [bos_id] + ids
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        y = _sample_tokens(model, x, max_tokens, eos_id)
        # decode bytes back to text, ignoring specials
        seq = y[0].tolist()
        if tokenizer_spec.get("add_special_tokens"):
            seq = [t for t in seq if t < 256]  # strip special tokens
        text = bytes(seq).decode("utf-8", errors="ignore")
        out[p] = text

    return out


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
