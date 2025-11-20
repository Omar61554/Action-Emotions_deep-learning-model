"""Minimal entry point for the Machine Vision project."""

import os
from . import utils


def main():
    print("Machine Vision project started.")
    here = os.path.dirname(__file__)
    print(f"Project src dir: {here}")
    # placeholder: load data, run pipeline


if __name__ == "__main__":
    main()
