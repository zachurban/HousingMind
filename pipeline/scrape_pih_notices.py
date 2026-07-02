#!/usr/bin/env python3
"""
scrape_pih_notices.py — HUD PIH Notices index scraper.

Fetches https://www.hud.gov/hudclips/notices/pih and produces pih_manifest.json:
for every notice listed (2020–present), extracts the number, title, PDF URLs,
and a structured parse of the status column (active / superseded / rescinded /
amended / expired), including cross-references to superseding notices.

Also compares against a local corpus folder (--corpus) and against the previous
manifest (--diff) to report what's new and what has changed.

Designed for GitHub Actions. Zero auth, minimal deps: requests + beautifulsoup4.

Usage:
    python scrape_pih_notices.py                            # write pih_manifest.json
    python scrape_pih_notices.py -o out.json                # custom output path
    python scrape_pih_notices.py --diff pih_manifest.json   # diff vs previous
    python scrape_pih_notices.py --corpus raw_documents/PIH_Notices_25_20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://www.hud.gov/hudclips/notices/pih"
HUD_BASE = "https://www.hud.gov"
USER_AGENT = "HousingMind-Scraper/1.0 (+https://housingmind.io)"

# --------------------------------------------------------------------------- #
# Parsing primitives
# --------------------------------------------------------------------------- #

# Notice-number reference inside status text. Handles:
#   "PIH 2025-16"    "PIH-2025-16"    "PIH_2025-16"
#   Optional joint prefix: "PIH 2025-03/H 2025-01" (we keep the PIH one)
#   Bare year-number after conjunction: "2021-38 and 2026-06"
NOTICE_REF_RE = re.compile(
    r"\b(?:PIH[\s\-_]*)?(\d{4}-\d{1,3})\b",
    re.IGNORECASE,
)

# Status events. Each keyword may appear once; we scan for all and merge.
EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("superseded_by", re.compile(r"superseded\s+by([^;.]+)", re.IGNORECASE)),
    ("rescinded_by",  re.compile(r"rescinded\s+by([^;.]+)", re.IGNORECASE)),
    ("amended_by",    re.compile(r"amended\s+by([^;.]+)", re.IGNORECASE)),
]
EXPIRED_RE = re.compile(
    r"expired\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})",
    re.IGNORECASE,
)

# Filenames in the local corpus, e.g. "PIH_2025-23.pdf", "2024-16pihn.pdf",
# "PIH-2026-06.pdf". Anchor on YYYY-NN pattern with 4-digit year 2000–2099.
FILENAME_NOTICE_RE = re.compile(r"(20\d{2}-\d{1,3})", re.IGNORECASE)

# Notice number in the "Document Number" cell (sometimes has "(Revised)" etc.)
DOC_NUMBER_RE = re.compile(r"^\s*(\d{4}-\d{1,3})\b\s*(.*)$")


def normalize_notice_id(raw: str) -> str:
    """Return canonical form 'PIH-YYYY-NN' from any of the observed variants."""
    m = NOTICE_REF_RE.search(raw)
    if not m:
        return raw.strip()
    return f"PIH-{m.group(1)}"


def parse_status(cell_text: str, cell_html) -> dict:
    """
    Turn the status cell into structured data.

    Returns:
        {
          "state": "active" | "superseded" | "rescinded" | "amended" | "expired",
          "raw":   original text,
          "events": [
            {"type": "superseded_by" | "rescinded_by" | "amended_by",
             "notices": ["PIH-2025-16", ...],
             "urls": ["https://..."]},
            {"type": "expired", "date": "December 31, 2022"}
          ]
        }
    """
    text = (cell_text or "").strip()
    result = {"state": "active", "raw": text, "events": []}

    if not text:
        return result

    # Collect any <a href="..."> URLs sitting inside the cell so we can attach
    # them to the events we detect.
    urls_in_cell: list[str] = []
    if cell_html is not None:
        for a in cell_html.find_all("a", href=True):
            urls_in_cell.append(_abs_url(a["href"]))

    # Extract each event type (may have multiple in one cell).
    for etype, pattern in EVENT_PATTERNS:
        for match in pattern.finditer(text):
            span = match.group(1)
            notices = [f"PIH-{n}" for n in NOTICE_REF_RE.findall(span)]
            if not notices:
                continue
            # URLs referenced immediately after the keyword: best-effort by
            # position — take any URLs from the cell that mention these notice
            # numbers in the href.
            matched_urls = [
                u for u in urls_in_cell
                if any(n.replace("PIH-", "") in u for n in notices)
            ]
            result["events"].append({
                "type": etype,
                "notices": notices,
                "urls": matched_urls,
            })

    m = EXPIRED_RE.search(text)
    if m:
        result["events"].append({"type": "expired", "date": m.group(1).strip()})

    # State precedence: rescinded > superseded > expired > amended > active
    types = {e["type"] for e in result["events"]}
    if "rescinded_by" in types:
        result["state"] = "rescinded"
    elif "superseded_by" in types:
        result["state"] = "superseded"
    elif "expired" in types:
        result["state"] = "expired"
    elif "amended_by" in types:
        result["state"] = "amended"  # still authoritative but modified
    return result


def _abs_url(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return HUD_BASE + href
    return href

# --------------------------------------------------------------------------- #
# Scrape
# --------------------------------------------------------------------------- #

@dataclass
class Notice:
    id: str                          # canonical "PIH-YYYY-NN"
    number: str                      # "YYYY-NN"
    year: int
    title: str
    revised: bool = False            # True if header showed "(Revised)"
    primary_url: Optional[str] = None
    attachment_urls: list[str] = field(default_factory=list)
    status: dict = field(default_factory=lambda: {"state": "active", "raw": "", "events": []})


def fetch_index_html(url: str = INDEX_URL) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_index(html: str) -> list[Notice]:
    """Parse every year table on the PIH notices index page."""
    soup = BeautifulSoup(html, "html.parser")
    notices: list[Notice] = []

    # Each year is an <h4> followed by a <table>. Find the h4s that contain a
    # four-digit year and walk to the next <table> sibling.
    for header in soup.find_all(["h4", "h3"]):
        year_text = header.get_text(strip=True)
        m = re.match(r"^(20\d{2})$", year_text)
        if not m:
            continue
        year = int(m.group(1))
        table = header.find_next("table")
        if table is None:
            continue
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            # Skip header rows
            if cells[0].find("strong") or cells[0].get_text(strip=True).lower().startswith("document"):
                continue
            notice = _parse_row(cells, year)
            if notice:
                notices.append(notice)
    return notices


def _parse_row(cells, year: int) -> Optional[Notice]:
    num_cell = cells[0]
    title_cell = cells[1]
    status_cell = cells[2] if len(cells) > 2 else None

    num_text = num_cell.get_text(" ", strip=True)
    dm = DOC_NUMBER_RE.match(num_text)
    if not dm:
        return None
    number = dm.group(1)
    trailing = dm.group(2) or ""
    revised = "revised" in trailing.lower()

    links = title_cell.find_all("a", href=True)
    pdf_links = [a for a in links if a["href"].lower().endswith(".pdf") or ".pdf" in a["href"].lower()]

    # Title = plain text of the title cell with link text preserved, minus URLs.
    title = title_cell.get_text(" ", strip=True)
    title = re.sub(r"\s+", " ", title).strip()

    primary = _abs_url(pdf_links[0]["href"]) if pdf_links else None
    attachments = [_abs_url(a["href"]) for a in pdf_links[1:]] if len(pdf_links) > 1 else []

    status_text = status_cell.get_text(" ", strip=True) if status_cell else ""
    status = parse_status(status_text, status_cell)

    return Notice(
        id=f"PIH-{number}",
        number=number,
        year=year,
        title=title,
        revised=revised,
        primary_url=primary,
        attachment_urls=attachments,
        status=status,
    )

# --------------------------------------------------------------------------- #
# Manifest I/O + comparisons
# --------------------------------------------------------------------------- #

def build_manifest(notices: list[Notice]) -> dict:
    counts = {"active": 0, "superseded": 0, "rescinded": 0, "amended": 0, "expired": 0}
    for n in notices:
        counts[n.status["state"]] = counts.get(n.status["state"], 0) + 1
    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source_url": INDEX_URL,
        "notice_count": len(notices),
        "counts_by_state": counts,
        "notices": {n.id: asdict(n) for n in sorted(
            notices, key=lambda x: (-x.year, -_num_key(x.number))
        )},
    }


def _num_key(number: str) -> int:
    """'2026-16' -> 16 for ordering within a year."""
    try:
        return int(number.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def diff_manifests(prev: dict, curr: dict) -> dict:
    prev_notices = prev.get("notices", {})
    curr_notices = curr.get("notices", {})
    added = sorted(set(curr_notices) - set(prev_notices))
    removed = sorted(set(prev_notices) - set(curr_notices))
    status_changed = []
    for nid, nc in curr_notices.items():
        np = prev_notices.get(nid)
        if not np:
            continue
        if nc["status"]["state"] != np["status"]["state"] or \
           nc["status"]["raw"] != np["status"]["raw"]:
            status_changed.append({
                "id": nid,
                "title": nc["title"],
                "old_state": np["status"]["state"],
                "old_raw": np["status"]["raw"],
                "new_state": nc["status"]["state"],
                "new_raw": nc["status"]["raw"],
            })
    return {"added": added, "removed": removed, "status_changed": status_changed}


def scan_corpus(corpus_dir: Path) -> set[str]:
    """Return set of canonical notice IDs found in a folder of PDFs."""
    ids: set[str] = set()
    if not corpus_dir.exists():
        return ids
    for pdf in corpus_dir.rglob("*.pdf"):
        m = FILENAME_NOTICE_RE.search(pdf.name)
        if m:
            ids.add(f"PIH-{m.group(1)}")
    return ids


def gaps_vs_corpus(manifest: dict, corpus_ids: set[str]) -> dict:
    online = set(manifest["notices"])
    missing_locally = sorted(online - corpus_ids)
    orphaned_locally = sorted(corpus_ids - online)  # in folder but not on index
    superseded_local = []
    for nid in sorted(corpus_ids & online):
        st = manifest["notices"][nid]["status"]
        if st["state"] in ("superseded", "rescinded"):
            superseded_local.append({
                "id": nid,
                "state": st["state"],
                "raw": st["raw"],
            })
    return {
        "missing_locally": missing_locally,
        "orphaned_locally": orphaned_locally,
        "superseded_local": superseded_local,
    }

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, default=Path("pih_manifest.json"))
    ap.add_argument("--url", default=INDEX_URL, help="Override index URL")
    ap.add_argument("--html", type=Path, help="Parse a local HTML file instead of fetching")
    ap.add_argument("--diff", type=Path, help="Compare against a previous manifest JSON")
    ap.add_argument("--corpus", type=Path, help="Local PDF folder to compare against")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.html:
        html = args.html.read_text()
    else:
        if not args.quiet:
            print(f"Fetching {args.url}...", file=sys.stderr)
        html = fetch_index_html(args.url)

    notices = parse_index(html)
    manifest = build_manifest(notices)

    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not args.quiet:
        print(f"Wrote {args.output} ({manifest['notice_count']} notices).", file=sys.stderr)
        print(f"  By state: {manifest['counts_by_state']}", file=sys.stderr)

    if args.diff and args.diff.exists():
        prev = load_manifest(args.diff)
        d = diff_manifests(prev, manifest)
        print("\n=== Diff vs previous manifest ===")
        print(f"Added:   {len(d['added'])}")
        for nid in d["added"][:20]:
            print(f"  + {nid} {manifest['notices'][nid]['title'][:70]}")
        if len(d["added"]) > 20:
            print(f"  ... and {len(d['added']) - 20} more")
        print(f"Removed: {len(d['removed'])}")
        for nid in d["removed"]:
            print(f"  - {nid}")
        print(f"Status changed: {len(d['status_changed'])}")
        for c in d["status_changed"][:20]:
            print(f"  ~ {c['id']}: {c['old_state']} -> {c['new_state']}")
            if c["new_raw"]:
                print(f"      {c['new_raw']}")
        if len(d["status_changed"]) > 20:
            print(f"  ... and {len(d['status_changed']) - 20} more")

    if args.corpus:
        corpus_ids = scan_corpus(args.corpus)
        g = gaps_vs_corpus(manifest, corpus_ids)
        print(f"\n=== Corpus at {args.corpus} ===")
        print(f"Local files:     {len(corpus_ids)}")
        print(f"Missing locally: {len(g['missing_locally'])}")
        for nid in g["missing_locally"][:30]:
            title = manifest['notices'][nid]['title'][:70]
            print(f"  + {nid} {title}")
        if len(g["missing_locally"]) > 30:
            print(f"  ... and {len(g['missing_locally']) - 30} more")
        print(f"\nSuperseded/rescinded in your corpus: {len(g['superseded_local'])}")
        for s in g["superseded_local"][:30]:
            print(f"  ! {s['id']} [{s['state']}] {s['raw']}")
        if len(g["superseded_local"]) > 30:
            print(f"  ... and {len(g['superseded_local']) - 30} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
