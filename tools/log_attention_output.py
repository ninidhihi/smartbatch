#!/usr/bin/env python3
"""Capture SmartBatch Arduino predictions and save CSV/XLSX results.

The sketch prints lines like:
attentive=0.812 drowsy=0.133 fidgeting=0.055 -> attentive | attention_score=81

This tool records the raw serial text and extracts those prediction rows into
spreadsheet-friendly outputs.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape


PREDICTION_RE = re.compile(
    r"attentive=(?P<attentive>-?\d+(?:\.\d+)?)\s+"
    r"drowsy=(?P<drowsy>-?\d+(?:\.\d+)?)\s+"
    r"fidgeting=(?P<fidgeting>-?\d+(?:\.\d+)?)\s+"
    r"->\s+(?P<prediction>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+\|\s+attention_score=(?P<attention_score>\d+))?"
)

HEADERS = [
    "timestamp",
    "elapsed_seconds",
    "attentive_score",
    "drowsy_score",
    "fidgeting_score",
    "prediction",
    "attention_score",
    "raw_line",
]


def parse_prediction(line: str, started_at: float) -> dict[str, object] | None:
    match = PREDICTION_RE.search(line)
    if not match:
        return None

    now = time.time()
    attention_score = match.group("attention_score")
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(now - started_at, 3),
        "attentive_score": float(match.group("attentive")),
        "drowsy_score": float(match.group("drowsy")),
        "fidgeting_score": float(match.group("fidgeting")),
        "prediction": match.group("prediction"),
        "attention_score": int(attention_score) if attention_score is not None else "",
        "raw_line": line.strip(),
    }


def iter_input_lines(args: argparse.Namespace):
    if args.input_log:
        with args.input_log.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle
        return

    if not args.port:
        for line in sys.stdin:
            yield line
        return

    try:
        import serial
    except ImportError as exc:
        raise SystemExit(
            "Serial capture requires pyserial. Install it with: python -m pip install pyserial"
        ) from exc

    with serial.Serial(args.port, args.baud, timeout=1) as connection:
        while True:
            data = connection.readline()
            if data:
                yield data.decode("utf-8", errors="replace")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_xml(row_index: int, col_index: int, value: object) -> str:
    ref = f"{column_name(col_index)}{row_index}"
    if isinstance(value, int | float) and value != "":
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    table = [HEADERS] + [[row.get(header, "") for header in HEADERS] for row in rows]
    sheet_rows = []
    for row_index, row in enumerate(table, start=1):
        cells = "".join(
            cell_xml(row_index, col_index, value)
            for col_index, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_index}">{cells}</row>')

    sheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:H{max(1, len(table))}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>
    <col min="1" max="1" width="21" customWidth="1"/>
    <col min="2" max="7" width="16" customWidth="1"/>
    <col min="8" max="8" width="72" customWidth="1"/>
  </cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
</worksheet>
"""

    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Attention Results" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

    workbook_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log SmartBatch serial predictions to raw text, CSV, and XLSX."
    )
    parser.add_argument("--port", help="Arduino serial port, for example COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, help="Capture duration in seconds")
    parser.add_argument("--input-log", type=Path, help="Parse an existing text log")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "attention_runs")
    parser.add_argument("--prefix", default="smartbatch_attention")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output_dir / f"{args.prefix}_{stamp}"
    raw_path = base.with_suffix(".txt")
    csv_path = base.with_suffix(".csv")
    xlsx_path = base.with_suffix(".xlsx")

    rows: list[dict[str, object]] = []
    started_at = time.time()
    deadline = started_at + args.duration if args.duration else None

    try:
        with raw_path.open("w", encoding="utf-8") as raw:
            for line in iter_input_lines(args):
                print(line.rstrip())
                raw.write(line)
                raw.flush()

                row = parse_prediction(line, started_at)
                if row is not None:
                    rows.append(row)
                    write_csv(csv_path, rows)

                if deadline is not None and time.time() >= deadline:
                    break
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        write_csv(csv_path, rows)
        write_xlsx(xlsx_path, rows)
        print(f"Raw log: {raw_path.resolve()}")
        print(f"CSV results: {csv_path.resolve()}")
        print(f"Excel results: {xlsx_path.resolve()}")
        print(f"Prediction rows saved: {len(rows)}")


if __name__ == "__main__":
    main()
