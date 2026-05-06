#!/usr/bin/env python3
"""Convenience script — equivalent to `python -m fmacnn [args]`."""
import sys
from pathlib import Path

# Allow running the script directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmacnn.cli import main

if __name__ == "__main__":
    main()
