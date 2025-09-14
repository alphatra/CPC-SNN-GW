"""
Deprecated HPO wrapper. Moved to examples/hpo.py
"""

import warnings

warnings.warn(
    "training.hpo_optuna is deprecated; use examples.hpo instead",
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name):  # lazy redirect to examples
    if name in {"run_hpo", "objective"}:
        from examples.hpo import run_hpo, objective  # type: ignore
        return {"run_hpo": run_hpo, "objective": objective}[name]
    raise AttributeError(name)


