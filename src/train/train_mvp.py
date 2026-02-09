"""
Deprecated MVP entrypoint.

This script used an older baseline architecture that is no longer maintained.
Use the canonical CPC-SNN training path:
    python -m src.train.train_cpc [args]
"""


def main() -> None:
    raise SystemExit(
        "train_mvp.py is deprecated. Use: python -m src.train.train_cpc --help"
    )


if __name__ == "__main__":
    main()
