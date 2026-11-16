"""Entry point of the ablation suite: ``python scripts/run_ablation.py --help``. All logic lives in the ``topobake`` package."""

import os
import sys

# make the package importable without installing it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topobake.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
