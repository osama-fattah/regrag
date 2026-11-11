"""A structured data source next to the documents: a CAPA register (CSV).

Answering "which critical CAPAs are overdue?" needs a query over records, not retrieval over text.
Giving the agent both lets it combine them ("CAPA-2026-007 is open; what does the SOP require?").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

FIELDS = (
    "capa_id",
    "deviation_id",
    "title",
    "classification",
    "status",
    "owner_area",
    "opened",
    "due",
    "effectiveness_check_due",
    "procedure",
)


def is_overdue(row: dict, as_of: date) -> bool:
    """Open CAPAs are judged against their due date; CAPAs in effectiveness check against the
    effectiveness-check date; closed CAPAs are never overdue."""
    if row["status"] == "open":
        return date.fromisoformat(row["due"]) < as_of
    if row["status"] == "effectiveness_check" and row.get("effectiveness_check_due"):
        return date.fromisoformat(row["effectiveness_check_due"]) < as_of
    return False


@dataclass
class CapaRegister:
    rows: list[dict]

    @classmethod
    def load(cls, path: str | Path) -> CapaRegister:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [{k: (v or None) for k, v in r.items()} for r in csv.DictReader(f)]
        missing = set(FIELDS) - set(rows[0]) if rows else set()
        if missing:
            raise ValueError(f"CAPA register is missing columns: {sorted(missing)}")
        return cls(rows)

    def query(
        self,
        capa_id: str | None = None,
        deviation_id: str | None = None,
        status: str | None = None,
        classification: str | None = None,
        procedure: str | None = None,
        overdue_as_of: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Filter the register. All filters are optional and combined with AND.

        overdue_as_of: ISO date; keeps CAPAs that are overdue on that date (see is_overdue).
        """

        def eq(a: str | None, b: str | None) -> bool:
            return b is None or (a or "").lower() == b.lower()

        cutoff = date.fromisoformat(overdue_as_of) if overdue_as_of else None
        out = []
        for r in self.rows:
            if not (
                eq(r["capa_id"], capa_id)
                and eq(r["deviation_id"], deviation_id)
                and eq(r["status"], status)
                and eq(r["classification"], classification)
                and eq(r["procedure"], procedure)
            ):
                continue
            if cutoff and not is_overdue(r, cutoff):
                continue
            out.append(dict(r))
        return out[:limit]
