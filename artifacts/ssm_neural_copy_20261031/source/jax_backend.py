"""Configure user-facing JAX backend names before JAX is imported."""

import os


def configure_environment(requested: str) -> None:
    if requested.lower() in {"auto", "metal"}:
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORM_NAME"] = requested
        os.environ["JAX_PLATFORMS"] = requested
