"""
Deprecated MVP evaluation entrypoint.

Use canonical evaluation scripts instead:
    python -m src.evaluation.evaluate_snn --help
    python -m src.evaluation.evaluate_background --help
"""


def main() -> None:
    raise SystemExit(
        "eval_mvp.py is deprecated. Use: python -m src.evaluation.evaluate_snn --help"
    )


if __name__ == "__main__":
    main()
