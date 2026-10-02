"""Check the fixed Agent delivery directory."""

from __future__ import annotations

import sys

from .check import check_world
from .paths import SOLUTION_DIR, WORLD_DIR
from .solution import check_links, check_solution


def main() -> int:
    violations = check_links(WORLD_DIR) + check_links(SOLUTION_DIR) + check_world(WORLD_DIR) + check_solution(SOLUTION_DIR)
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("checker takes no arguments")
    raise SystemExit(main())
