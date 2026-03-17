from __future__ import annotations

import warnings


def configure_runtime_warnings() -> None:
    """Silence optional pandas accelerator warnings for CLI execution."""

    warnings.filterwarnings(
        "ignore",
        message=r"Pandas requires version .* of 'numexpr'.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Pandas requires version .* of 'bottleneck'.*",
        category=UserWarning,
    )
