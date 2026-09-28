"""Text-table rendering shared by the SAE evaluation modules.

Each evaluation prints a summary table and writes the same table to
``summary_table.txt``. Rendering once here keeps the two in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    n_key_cols: int = 1,
) -> str:
    """Render a fixed-width table; key columns left-aligned, the rest right-aligned."""
    if not rows:
        return ""
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]

    def render(cells: Sequence[str]) -> str:
        parts = [
            f"{str(cell):<{widths[i]}}" if i < n_key_cols else f"{str(cell):>{widths[i]}}"
            for i, cell in enumerate(cells)
        ]
        return " ".join(parts).rstrip()

    header_line = render(headers)
    lines = [header_line, "-" * len(header_line)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines) + "\n"


def write_table(output_dir: Path, table: str, name: str = "summary_table.txt") -> None:
    with open(output_dir / name, "w", encoding="utf-8") as f:
        f.write(table)
