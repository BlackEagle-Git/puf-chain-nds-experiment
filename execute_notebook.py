from __future__ import annotations

import argparse
import json
from pathlib import Path


def execute_notebook(notebook_path: Path) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__"}

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{notebook_path.name}#cell_{idx}", "exec"), scope, scope)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Notebook failed at code cell index {idx}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute .ipynb code cells sequentially.")
    parser.add_argument("notebook", type=str)
    args = parser.parse_args()
    path = Path(args.notebook).resolve()
    execute_notebook(path)
    print(f"Notebook executed successfully: {path}")


if __name__ == "__main__":
    main()
