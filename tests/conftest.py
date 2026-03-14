"""Pytest configuration: adds repo root and research sub-packages to sys.path so
test modules can import production code without install."""
import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESEARCH = os.path.join(_REPO_ROOT, "research")
_FEATURES = os.path.join(_RESEARCH, "features")
_TRANSFORMER = os.path.join(_RESEARCH, "transformer")

for _path in (_REPO_ROOT, _RESEARCH, _FEATURES, _TRANSFORMER):
    if _path not in sys.path:
        sys.path.insert(0, _path)
