from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from orya_guard.errors import OryaGuardError
from orya_guard.runtime import configure_runtime_warnings

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
    help=(
        "Validate datasets and inference payloads before training or serving.\n\n"
        "Use 'check-dataset' to inspect one dataset, 'compare' to compare a "
        "training dataset with a candidate dataset, and 'check-inference-payload' "
        "to validate a JSON payload against a schema."
    ),
)


def _write_markdown_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _handle_error(error: OryaGuardError) -> None:
    typer.echo(
        "\n".join(
            [
                "Status: FAIL",
                "",
                "What went wrong",
                "---------------",
                f"- {error.message}",
                "",
                "How to fix it",
                "-------------",
                f"- {error.next_step}",
            ]
        ),
        err=True,
    )
    raise typer.Exit(code=1) from error


@app.command(
    "check-dataset",
    help=(
        "Inspect one CSV or parquet dataset for common data quality issues.\n\n"
        "Reports row and column counts, column dtypes, duplicate rows, null ratios, "
        "and constant columns."
    ),
)
def check_dataset_command(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to the dataset file. Supported formats: .csv, .parquet."),
    ],
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            "-o",
            help="Write the results as a Markdown report to PATH.",
        ),
    ] = None,
) -> None:
    """Inspect a dataset and print a concise summary."""

    try:
        configure_runtime_warnings()
        from orya_guard.checks.dataset import check_dataset
        from orya_guard.io.data_loading import load_dataframe
        from orya_guard.reporting.console import render_dataset_summary
        from orya_guard.reporting.markdown import render_dataset_markdown

        dataframe = load_dataframe(dataset_path)
        result = check_dataset(dataframe, dataset_path)
        typer.echo(render_dataset_summary(result))
        if report_path is not None:
            _write_markdown_report(report_path, render_dataset_markdown(result))
            typer.echo(f"\nReport saved: {report_path}")
    except OryaGuardError as error:
        _handle_error(error)


@app.command(
    "compare",
    help=(
        "Compare a candidate dataset with a training reference dataset.\n\n"
        "Checks schema compatibility, missing or extra columns, null ratio changes, "
        "unseen categorical values, and simple numeric drift signals."
    ),
)
def compare_command(
    train_dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to the training reference dataset (.csv or .parquet)."),
    ],
    candidate_dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to the candidate dataset (.csv or .parquet)."),
    ],
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            "-o",
            help="Write the comparison results as a Markdown report to PATH.",
        ),
    ] = None,
) -> None:
    """Compare train and candidate datasets before training or rollout."""

    try:
        configure_runtime_warnings()
        from orya_guard.checks.comparison import compare_datasets
        from orya_guard.io.data_loading import load_dataframe
        from orya_guard.reporting.console import render_compare_summary
        from orya_guard.reporting.markdown import render_compare_markdown

        train_df = load_dataframe(train_dataset_path)
        candidate_df = load_dataframe(candidate_dataset_path)
        result = compare_datasets(
            train_df,
            candidate_df,
            train_dataset_path,
            candidate_dataset_path,
        )
        typer.echo(render_compare_summary(result))
        if report_path is not None:
            _write_markdown_report(report_path, render_compare_markdown(result))
            typer.echo(f"\nReport saved: {report_path}")
        if result.has_errors():
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except OryaGuardError as error:
        _handle_error(error)


@app.command(
    "check-inference-payload",
    help=(
        "Validate one JSON inference payload against a schema.\n\n"
        "Checks that required fields are present, unexpected fields are absent, "
        "and simple field types match the schema."
    ),
)
def check_inference_payload_command(
    payload_path: Annotated[
        Path,
        typer.Argument(help="Path to the JSON payload file to validate."),
    ],
    schema_path: Annotated[
        Path,
        typer.Option(
            "--schema",
            "-s",
            help="Path to the schema JSON file with a top-level 'fields' mapping.",
        ),
    ],
) -> None:
    """Validate an inference payload against a simple schema definition."""

    try:
        configure_runtime_warnings()
        from orya_guard.checks.inference import validate_inference_payload
        from orya_guard.io.data_loading import load_inference_schema, load_json_payload
        from orya_guard.reporting.console import render_inference_summary

        payload = load_json_payload(payload_path)
        payload_schema = load_inference_schema(schema_path)
        result = validate_inference_payload(payload, payload_schema, payload_path)
        typer.echo(render_inference_summary(result, schema_path=str(schema_path)))
        if result.has_errors():
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except OryaGuardError as error:
        _handle_error(error)


def main() -> None:
    """Run the CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
