"""
CLI Module: Command Line Interface Components

Modular implementation of CLI functionality split from
cli.py for better maintainability.

Subpackages:
- commands: Command implementations (train, eval, infer)
- parsers: Argument parsers and validation
- utils: CLI utilities and helpers
"""

# Keep package import light-weight to avoid import-time failures in tests
__all__: list[str] = []
