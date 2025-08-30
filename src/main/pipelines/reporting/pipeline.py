from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_and_consolidate,
    filter_metrics,
    plot_val_loss_vs_steps,
    plot_model_size_vs_final_ppl,
    plot_ema_vs_non_ema,
    plot_steps_saved,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_and_consolidate,
            inputs="params:reporting",
            outputs="reporting.metrics_consolidated_df",
            name="load_and_consolidate",
        ),
        node(
            func=lambda df: df,
            inputs="reporting.metrics_consolidated_df",
            outputs="reporting.metrics_consolidated",
            name="save_consolidated_metrics",
        ),
        node(
            func=filter_metrics,
            inputs=dict(metrics="reporting.metrics_consolidated_df",
                        params="params:reporting"),
            outputs="reporting.filtered_metrics",
            name="filter_metrics",
        ),
        node(
            func=plot_val_loss_vs_steps,
            inputs=dict(df="reporting.filtered_metrics",
                        params="params:reporting",
                        out_path="params:reporting.output_paths.val_loss"),
            outputs="val_loss_plot_path",
            name="plot_val_loss",
        ),
        node(
            func=plot_model_size_vs_final_ppl,
            inputs=dict(df="reporting.filtered_metrics",
                        params="params:reporting",
                        out_path="params:reporting.output_paths.size_vs_ppl"),
            outputs="size_vs_ppl_plot_path",
            name="plot_size_vs_ppl",
        ),
        node(
            func=plot_ema_vs_non_ema,
            inputs=dict(df="reporting.filtered_metrics",
                        params="params:reporting",
                        out_path="params:reporting.output_paths.ema_vs_non_ema"),
            outputs="ema_vs_nonema_plot_path",
            name="plot_ema_vs_nonema",
        ),
        node(
            func=plot_steps_saved,
            inputs=dict(df="reporting.filtered_metrics",
                        params="params:reporting",
                        out_path="params:reporting.output_paths.steps_saved"),
            outputs="steps_saved_plot_path",
            name="plot_steps_saved",
        ),
    ])