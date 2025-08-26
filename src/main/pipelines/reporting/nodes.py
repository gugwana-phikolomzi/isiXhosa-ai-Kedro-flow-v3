from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Styling (monotone / paper)
# ----------------------------

@contextmanager
def paper_style():
    """Monotone, minimalist research-paper aesthetic."""
    with plt.rc_context({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
        "lines.linewidth": 1.6,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    }):
        yield


# For multi-line plots: use black with different dashes
_LINE_STYLES = ["-", "--", "-.", ":"]
_BLACK = "#000000"
_DARK_GRAY = "#555555"
_GRAY = "#888888"
_LIGHT_GRAY = "#BBBBBB"


@dataclass
class ReportingParams:
    include_experiments: List[str]
    max_steps: int
    approx_params: Dict[str, int]
    ema_pairs: List[Tuple[str, str]]
    metrics_root: str
    output_paths: Dict[str, str]


def _coerce_params(p: Any) -> ReportingParams:
    if isinstance(p, ReportingParams):
        return p
    if isinstance(p, dict):
        allowed_keys = {
            "include_experiments",
            "max_steps",
            "metrics_root",
            "approx_params",
            "ema_pairs",
            "output_paths",
        }
        kw = {k: v for k, v in p.items() if k in allowed_keys}
        return ReportingParams(**kw)
    raise TypeError("reporting parameters must be a dict or ReportingParams")


def load_and_consolidate(params: Any) -> pd.DataFrame:
    """Glob data/08_reporting/*/metrics.csv and return one tidy DF."""
    params = _coerce_params(params)
    root = Path(params.metrics_root)
    paths = sorted(root.glob("*/metrics.csv"))
    frames = []
    for path in paths:
        exp_id = path.parent.name  # A0, A1, ...
        df = pd.read_csv(path)
        if "experiment_id" not in df.columns:
            df["experiment_id"] = exp_id
        else:
            df["experiment_id"] = df["experiment_id"].fillna(exp_id).replace("", exp_id)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No metrics.csv files under {root}/*/")
    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["experiment_id", "step"])
        .reset_index(drop=True)
    )
    return out


def filter_metrics(metrics: pd.DataFrame, params: Any) -> pd.DataFrame:
    params = _coerce_params(params)
    df = metrics.copy()
    return df[df["experiment_id"].isin(params.include_experiments)].sort_values(
        ["experiment_id", "step"]
    )


def _last_row_per_experiment(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("step").groupby("experiment_id", as_index=False).tail(1)


# ---------------- Plots (monotone) ---------------- #

def plot_val_loss_vs_steps(df: pd.DataFrame, params: Any, out_path: str) -> str:
    params = _coerce_params(params)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with paper_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for i, (eid, g) in enumerate(df.groupby("experiment_id")):
            g = g.sort_values("step")
            ls = _LINE_STYLES[i % len(_LINE_STYLES)]
            ax.plot(g["step"], g["val_loss"], linestyle=ls, color=_BLACK, label=eid)
            # early-stopping marker at last step
            last_step = int(g["step"].iloc[-1])
            ax.axvline(last_step, linestyle=":", color=_GRAY, linewidth=1)

        # Minimalist axes
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.set_xlabel("Steps")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation loss vs steps (dashed line = early stop)")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    return str(out)


def plot_model_size_vs_final_ppl(df: pd.DataFrame, params: Any, out_path: str) -> str:
    params = _coerce_params(params)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    last = _last_row_per_experiment(df)
    last["params"] = last["experiment_id"].map(params.approx_params)
    last = last.dropna(subset=["params", "ppl_v"])
    agg = last.groupby(["experiment_id", "params"], as_index=False)["ppl_v"].min()

    with paper_style():
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.scatter(agg["params"], agg["ppl_v"], color=_BLACK, s=18, zorder=3)

        # Labels near points
        for _, r in agg.iterrows():
            ax.annotate(
                r["experiment_id"],
                (r["params"], r["ppl_v"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=9,
                color=_DARK_GRAY,
            )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.set_xscale("log")
        ax.set_xlabel("Parameters (log scale)")
        ax.set_ylabel("Final validation perplexity")
        ax.set_title("Model size vs final perplexity")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    return str(out)


def plot_ema_vs_non_ema(df: pd.DataFrame, params: Any, out_path: str) -> str:
    params = _coerce_params(params)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    last = _last_row_per_experiment(df).set_index("experiment_id")
    labels, noema, ema = [], [], []
    for a, b in params.ema_pairs:
        if a in last.index and b in last.index:
            labels.append(f"{a}→{b}")
            noema.append(float(last.loc[a, "val_loss"]))
            ema.append(float(last.loc[b, "val_loss"]))

    with paper_style():
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        if labels:
            x = np.arange(len(labels))
            w = 0.36
            ax.bar(x - w / 2, noema, width=w, color=_LIGHT_GRAY, edgecolor=_BLACK, label="Non-EMA")
            ax.bar(x + w / 2, ema,   width=w, color=_BLACK,      edgecolor=_BLACK, label="EMA")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.set_ylabel("Final validation loss")
            ax.set_title("EMA vs Non-EMA")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid EMA pairs", ha="center", va="center", color=_DARK_GRAY)
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    return str(out)


def plot_steps_saved(df: pd.DataFrame, params: Any, out_path: str) -> str:
    params = _coerce_params(params)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    last = _last_row_per_experiment(df)
    last["saved"] = params.max_steps - last["step"]
    last = last.sort_values("saved", ascending=False)

    with paper_style():
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.bar(last["experiment_id"], last["saved"], color=_BLACK, edgecolor=_BLACK)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_ylabel(f"Steps saved (of {params.max_steps})")
        ax.set_title("Early stopping: steps saved")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    return str(out)
