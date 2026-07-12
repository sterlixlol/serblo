import sys, pathlib

# src-layout: make `serblo` importable without an editable install
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
