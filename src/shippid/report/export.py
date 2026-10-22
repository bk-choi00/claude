"""CSV (Excel-friendly, UTF-8 with BOM) and JSON export of an analysis report."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..model import AnalysisReport

# Cells starting with these characters are interpreted as formulas by spreadsheet programs
# (CSV injection); such text cells are prefixed with an apostrophe.
_FORMULA_START = ("=", "+", "@", "\t", "\r")
_NUMBER_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, (list, tuple, set)):
        v = "; ".join(str(x) for x in v)
    elif isinstance(v, dict):
        v = json.dumps(v, ensure_ascii=False, default=str)
    else:
        v = str(getattr(v, "value", v))
    if v.startswith(_FORMULA_START) or (v.startswith("-") and not _NUMBER_RE.match(v)):
        v = "'" + v
    return v


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path, columns: list[str] | None = None) -> Path:
    """Write dict rows as CSV (UTF-8 with BOM so Excel detects the encoding); the columns are
    ``columns`` or the union of the row keys in first-seen order."""
    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(k for r in rows for k in r))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in columns])
    return path


def export_tables(report: AnalysisReport, out_dir: str | Path, prefix: str = "") -> dict[str, Path]:
    """Write ``equipment / valves / instruments / lines / findings / flow_paths`` CSV files and the
    full JSON report into ``out_dir``; returns ``{name: path}`` (``report_json`` for the JSON)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    tables = {
        "equipment": report.equipment_list,
        "valves": report.valve_list,
        "instruments": report.instrument_list,
        "lines": report.line_list,
    }
    for name, rows in tables.items():
        out[name] = write_csv(rows, out_dir / f"{prefix}{name}.csv")
    finding_cols = ["id", "rule_id", "verdict", "severity", "title", "description", "object_ids", "reference",
                    "source", "confidence"]
    out["findings"] = write_csv((f.model_dump(mode="json") for f in report.findings), out_dir / f"{prefix}findings.csv",
                                finding_cols)
    out["flow_paths"] = write_csv(
        ({"n": k, "source_id": fp.source_id, "target_id": fp.target_id, "description": fp.description or "",
          "node_ids": fp.node_ids, "pipe_ids": fp.pipe_ids} for k, fp in enumerate(report.flow_paths, 1)),
        out_dir / f"{prefix}flow_paths.csv", ["n", "source_id", "target_id", "description", "node_ids", "pipe_ids"])
    out["report_json"] = report.save(out_dir / f"{prefix}report.json")
    return out
