from __future__ import annotations

from kedro.pipeline import Pipeline, pipeline

from .pipelines import (
    model_A0,
    model_A1,
    model_A2,
    model_A3,
    model_A4,
    model_A5,
    model_A6,
    model_A7,
    model_B0,
    model_B1,
    model_B2,
    model_B21,
    model_b211,
    model_b212,
    model_b213,
    model_b214,
    model_B22,
    model_C0,
    model_C1,

    model_D0,
    reporting,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipelines: dict[str, Pipeline] = {
        "model_A0": model_A0.create_pipeline(),
        "model_A1": model_A1.create_pipeline(),
        "model_A2": model_A2.create_pipeline(),
        "model_A3": model_A3.create_pipeline(),
        "model_A4": model_A4.create_pipeline(),
        "model_A5": model_A5.create_pipeline(),
        "model_A6": model_A6.create_pipeline(),
        "model_A7": model_A7.create_pipeline(),
        "model_B0": model_B0.create_pipeline(),
        "model_B1": model_B1.create_pipeline(),
        "model_B2": model_B2.create_pipeline(),
        "model_B21": model_B21.create_pipeline(),
        "model_b211": model_b211.create_pipeline(),
        "model_b212": model_b212.create_pipeline(),
        "model_b213": model_b213.create_pipeline(),
        "model_b214": model_b214.create_pipeline(),
        "model_B22": model_B22.create_pipeline(),
        "model_C0": model_C0.create_pipeline(),
        "model_C1": model_C1.create_pipeline(),

        "model_D0":model_D0.create_pipeline(),
        "reporting": reporting.create_pipeline(),
    }

    pipelines["__default__"] = pipelines["model_A0"]

    return pipelines
