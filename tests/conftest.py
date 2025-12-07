import sys
from pathlib import Path

# Ensure repository root is on sys.path when pytest runs so local package imports work.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
import sys

# Ensure the repository root (project) is on sys.path when running pytest so
# tests can import the `HMM` package even if the project isn't installed.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
