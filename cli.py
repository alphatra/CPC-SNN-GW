#!/usr/bin/env python3
"""
Main CLI entry point for CPC+SNN Neuromorphic GW Detection.

This is a thin wrapper that delegates to the modular CLI structure.
The actual implementation is in the cli/ folder for better organization.
"""

import sys
from cli.main import main

if __name__ == "__main__":
    sys.exit(main())
