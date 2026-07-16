"""
DEPRECATED. The single canonical, governed entry point is now :mod:`crashsev.cli`.

The original ``run_benchmark`` executed load -> map -> split -> fit -> final-evaluate in one
shot, with no development/final-test separation, no lock, and overwriteable bundles (AA2-007/
008/017). Those governance requirements are met only by the phased CLI:

    python -m crashsev.cli validate-data     --data <file>
    python -m crashsev.cli develop           --data <file> --config configs/route_a_09_12.yml
    python -m crashsev.cli freeze-experiment
    python -m crashsev.cli evaluate-final     --data <file> --config configs/route_a_09_12.yml

This module remains only so old references fail loudly with that guidance.
"""
from __future__ import annotations

import sys


def main(argv=None):
    sys.stderr.write(__doc__)
    raise SystemExit(
        "crashsev.run_benchmark is deprecated; use `python -m crashsev.cli <phase>` "
        "(validate-data | develop | freeze-experiment | evaluate-final)."
    )


if __name__ == "__main__":
    main()
