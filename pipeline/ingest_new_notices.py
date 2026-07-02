#!/usr/bin/env python3
"""
ingest_new_notices.py — Turn manifest deltas into vector-ready chunks.

Reads pipeline/pih_manifest.json, compares to pipeline/ingested_state.json,
downloads each not-yet-ingested active notice's PDF, extracts + chunks +
embeds, and pushes cumulative chunks + embeddings to a private HF Dataset
(default: Zachurban/housingmind-updates).

The Space's app.py pulls those artifacts at startup and merges them into
its local ChromaDB. Delta-only chunks — the base 74K stays untouched.

Requires: PyMuPDF, sentence-transformers, huggingface_hub, requests, numpy.
Env: HF_TOKEN (with write access to the target dataset).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_MANIFEST = Path("pipeline/pih_manifest.json")
DEFAULT_STATE = Path("pipeline/ingested_state.json")
DEFAULT_DATASET = os.environ.get("HOUSINGMIND_DATASET", "Zachurban/housingmind-updates")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
USER_AGENT = "HousingMind-Ingester/1.0"
CHUNK_TARGET = 1200
CHUNK_OVERLAP = 150
DOWNLOAD_TIMEOUT = 60

# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #

def download_pdf(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    return r.content


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, 1):
        text = page.get_text() or ""
        pages.append((i, text))
    doc.close()
    return pages

# --------------------------------------------------------------------------- #
# Chunking — paragraph-aware, tracks pages, aims for CHUNK_TARGET chars.
# Not fancy but consistent with how the base corpus was chunked.
# --------------------------------------------------------------------------- #

PARA_RE = re.compile(r"\n\s*\n")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
SECTION_HINT_RE = re.compile(
    r"^\s*(?:SECTION|Section|(?:[IVX]+|\d+)\.)\s+[A-Z0-9]", re.MULTILINE
)


def _split_long_paragraph(para: str, target: int) -> list[str]:
    """If a paragraph exceeds target*1.5, split on sentence boundaries."""
    if len(para) <= int(target * 1.5):
        return [para]
    pieces, buf = [], ""
    for sent in SENTENCE_RE.split(para):
        if len(buf) + len(sent) + 1 > target and buf:
            pieces.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}" if buf else sent
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def chunk_pages(pages: list[tuple[int, str]],
                target: int = CHUNK_TARGET,
                overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split pages into chunks. Each chunk carries its starting page."""
    chunks: list[dict] = []
    buf, buf_page = "", None

    for page_num, text in pages:
        text = re.sub(r"[ \t]+\n", "\n", text).strip()
        if not text:
            continue
        for para in PARA_RE.split(text):
            para = para.strip()
            if not para:
                continue
            for piece in _split_long_paragraph(para, target):
                if buf_page is None:
                    buf_page = page_num
                if len(buf) + len(piece) + 2 > target and buf:
                    chunks.append({"page": buf_page, "text": buf.strip()})
                    tail = buf[-overlap:] if len(buf) > overlap else ""
                    buf, buf_page = (tail + "\n\n" + piece), page_num
                else:
                    buf = f"{buf}\n\n{piece}" if buf else piece

    if buf.strip():
        chunks.append({"page": buf_page or 1, "text": buf.strip()})

    return chunks

# --------------------------------------------------------------------------- #
# Topic classification — mirrors the base corpus's schema (HCV / PH / etc.)
# --------------------------------------------------------------------------- #

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "HCV":            ["housing choice voucher", "hcv program", "voucher program",
                       "tenant-based", "section 8 tenant"],
    "Public Housing": ["public housing", "operating fund", "capital fund",
                       "pha operating", "phas ", "acc "],
    "RAD":            ["rental assistance demonstration", "rad conversion"],
    "PBV":            ["project-based voucher", "pbv "],
    "MTW":            ["moving to work", "mtw demonstration", "mtw agency"],
    "NSPIRE":         ["nspire", "national standards for the physical inspection"],
    "HOTMA":          ["hotma", "housing opportunity through modernization"],
    "EIV":            ["enterprise income verification", "eiv system"],
    "FSS":            ["family self-sufficiency", "fss program"],
    "Section 3":      ["section 3 ", "24 cfr part 75"],
    "Section 18":     ["section 18 ", "disposition application", "demolition and disposition"],
    "Fair Housing":   ["fair housing", "affirmatively furthering"],
    "Tribal":         ["nahasda", "indian housing", "tribal ", "native american housing"],
    "Emergency":      ["emergency housing voucher", "ehv ", "stability voucher"],
}


def classify_topic(text: str) -> str:
    haystack = text[:8000].lower()
    scores = {t: sum(haystack.count(kw) for kw in kws) for t, kws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Cross-cutting"

# --------------------------------------------------------------------------- #
# State I/O
# --------------------------------------------------------------------------- #

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"dataset_repo": DEFAULT_DATASET, "ingested_notices": {}}


def save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2))

# --------------------------------------------------------------------------- #
# HF Dataset I/O
# --------------------------------------------------------------------------- #

def fetch_existing_dataset(repo_id: str, token: Optional[str]) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Return (existing_chunks, existing_embeddings_by_id). Empty if dataset is new."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    chunks: list[dict] = []
    embeddings: dict[str, np.ndarray] = {}

    for filename, target in [("chunks.jsonl", "chunks"), ("embeddings.npz", "embeddings")]:
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=filename,
                repo_type="dataset", token=token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue
        except Exception as e:
            print(f"  Warning fetching {filename}: {e}")
            continue
        if target == "chunks":
            with open(path) as f:
                chunks = [json.loads(line) for line in f if line.strip()]
        else:
            arr = np.load(path)
            embeddings = {i: v for i, v in zip(arr["ids"], arr["vectors"])}

    print(f"  Existing dataset: {len(chunks)} chunks, {len(embeddings)} embeddings")
    return chunks, embeddings


def push_dataset(repo_id: str, token: str,
                 chunks: list[dict], embeddings_by_id: dict[str, np.ndarray],
                 dry_run: bool = False) -> None:
    """Write cumulative chunks + embeddings to the HF dataset (private)."""
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        chunks_path = td / "chunks.jsonl"
        emb_path = td / "embeddings.npz"
        meta_path = td / "metadata.json"

        with open(chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        ids = list(embeddings_by_id)
        vectors = np.array([embeddings_by_id[i] for i in ids], dtype=np.float32)
        np.savez_compressed(emb_path, ids=np.array(ids), vectors=vectors)

        notice_ids = sorted({c["metadata"]["notice_id"] for c in chunks})
        meta_path.write_text(json.dumps({
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
            "embedding_dim": int(vectors.shape[1]) if len(vectors) else 0,
            "embed_model": EMBED_MODEL,
            "notice_count": len(notice_ids),
            "notices": notice_ids,
        }, indent=2))

        if dry_run:
            print("  [dry-run] would upload:")
            for p in (chunks_path, emb_path, meta_path):
                print(f"    {p.name} ({p.stat().st_size} bytes)")
            return

        api = HfApi(token=token)
        # Create dataset repo if missing; private by default
        api.create_repo(
            repo_id=repo_id, repo_type="dataset",
            private=True, exist_ok=True,
        )
        for p in (chunks_path, emb_path, meta_path):
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=repo_id, repo_type="dataset",
                commit_message=f"Update {p.name}",
            )
        print(f"  Pushed to {repo_id}: {len(chunks)} chunks, {len(vectors)} embeddings.")

# --------------------------------------------------------------------------- #
# Notice selection
# --------------------------------------------------------------------------- #

def select_new_notices(manifest: dict, state: dict, limit: Optional[int]) -> list[dict]:
    """Return active notices in the manifest not yet in state, oldest-first."""
    ingested = set(state.get("ingested_notices", {}))
    pending = []
    for nid, n in manifest.get("notices", {}).items():
        if nid in ingested:
            continue
        if n.get("status", {}).get("state") != "active":
            continue
        if not n.get("primary_url"):
            continue
        pending.append(n)

    # Newest first — 2026-16 before 2025-30 — but let the CLI reverse if needed.
    pending.sort(key=lambda n: (n["year"], _num_key(n["number"])), reverse=True)
    if limit:
        pending = pending[:limit]
    return pending


def _num_key(number: str) -> int:
    try:
        return int(number.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0

# --------------------------------------------------------------------------- #
# Per-notice processing
# --------------------------------------------------------------------------- #

def process_notice(notice: dict) -> tuple[list[dict], list[str]]:
    """Download, extract, chunk, tag. Returns (chunks, texts_in_order)."""
    nid = notice["id"]                              # e.g. "PIH-2025-30"
    number = notice["number"]                       # e.g. "2025-30"
    url = notice["primary_url"]
    title = notice.get("title", "")

    print(f"  {nid}: fetching {url}")
    pdf = download_pdf(url)
    pages = extract_pages(pdf)
    if not pages:
        print(f"  {nid}: no text extracted, skipping.")
        return [], []

    full_text = "\n".join(t for _, t in pages)
    topic = classify_topic(full_text)
    print(f"  {nid}: {len(pages)} pages, topic={topic}")

    raw_chunks = chunk_pages(pages)
    source_name = Path(url).name
    citation = f"PIH {number}"

    out, texts = [], []
    for i, ch in enumerate(raw_chunks, 1):
        chunk_id = f"{nid.lower()}-c{i:03d}"
        out.append({
            "chunk_id": chunk_id,
            "text": ch["text"],
            "metadata": {
                "source": source_name,
                "document_type": "PIH Notice",
                "topic": topic,
                "notice_number": number,
                "notice_id": nid,
                "citation": citation,
                "title": title,
                "page": ch["page"],
                "chunk_index": i,
                "primary_url": url,
            },
        })
        texts.append(ch["text"])
    print(f"  {nid}: {len(out)} chunks")
    return out, texts

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, help="Max new notices to process this run")
    ap.add_argument("--dry-run", action="store_true", help="Do not push to HF or update state")
    ap.add_argument("--only", nargs="+", help="Explicit notice IDs like PIH-2025-30 (ignores state)")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text())
    state = load_state(args.state)
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN not set — either export it or use --dry-run.", file=sys.stderr)
        return 2

    # Selection
    if args.only:
        wanted = set(args.only)
        pending = [n for nid, n in manifest["notices"].items() if nid in wanted]
    else:
        pending = select_new_notices(manifest, state, args.limit)

    if not pending:
        print("No new active notices to ingest.")
        return 0

    print(f"\n== Ingesting {len(pending)} notice(s) ==")
    all_new_chunks: list[dict] = []
    all_new_texts: list[str] = []

    for n in pending:
        try:
            chunks, texts = process_notice(n)
            if chunks:
                all_new_chunks.extend(chunks)
                all_new_texts.extend(texts)
                state.setdefault("ingested_notices", {})[n["id"]] = {
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": len(chunks),
                    "primary_url": n["primary_url"],
                    "title": n.get("title", ""),
                }
        except Exception as e:
            print(f"  {n['id']}: FAILED ({e}); leaving unstate for retry.")

    if not all_new_chunks:
        print("Nothing usable produced.")
        return 1

    # Embed only the new chunks; merge with existing dataset
    print(f"\n== Embedding {len(all_new_chunks)} new chunks ==")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    new_vectors = model.encode(
        all_new_texts, batch_size=16,
        show_progress_bar=True, normalize_embeddings=True,
    )

    print(f"\n== Merging with existing dataset {args.dataset} ==")
    existing_chunks, existing_embs = ([], {}) if args.dry_run else fetch_existing_dataset(args.dataset, token)
    existing_ids = {c["chunk_id"] for c in existing_chunks}

    for c, v in zip(all_new_chunks, new_vectors):
        if c["chunk_id"] not in existing_ids:
            existing_chunks.append(c)
        existing_embs[c["chunk_id"]] = np.asarray(v, dtype=np.float32)

    print(f"  Total after merge: {len(existing_chunks)} chunks")

    print(f"\n== Pushing to {args.dataset} ==")
    push_dataset(args.dataset, token, existing_chunks, existing_embs, dry_run=args.dry_run)

    if not args.dry_run:
        save_state(state, args.state)
        print(f"  State written to {args.state}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
