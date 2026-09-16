"""Convenience dispatcher: `python main.py <bench|sweep|aggregate> [args...]`."""

import sys

COMMANDS = {"bench": "scaling.benchmark", "sweep": "scaling.sweep",
            "aggregate": "scaling.aggregate", "plot": "scaling.plot"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: python main.py {{{'|'.join(COMMANDS)}}} [args...]")
        return 1
    import importlib

    module = importlib.import_module(COMMANDS[sys.argv[1]])
    return module.main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
