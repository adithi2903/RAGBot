"""
Build the search index from your PDFs (SECTION-AWARE labelling):

    python ingest.py

Chunks are sized normally (so the count stays sensible), but each chunk is
LABELLED with the legal section it falls under (e.g. "Sec 80C") by detecting
section headers in the text. That gives exact citations without over-splitting.
Re-run whenever you add or change PDFs.
"""
import os
# Use locally-cached models; don't ping Hugging Face every run (it can hang).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
import glob
import bisect
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

from config import PDF_DIR, CHROMA_DIR, COLLECTION, EMB_MODEL, CHUNK_SIZE, OVERLAP

# A section header like "80C.", "139.", "115BAC." followed by a capitalised heading.
SECTION_RE = re.compile(r'(?<![\dA-Za-z(.])(\d{1,3}[A-Z]{0,3})\.\s+(?=[A-Z])')


def parse_filename(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    doc_type = parts[0] if parts else "Unknown"
    year = next((p for p in parts if p.isdigit() and len(p) == 4), "")
    is_current = doc_type.lower().startswith("itact") and year == "2025"
    return {
        "source_file": os.path.basename(path),
        "doc_type": doc_type,
        "year": year,
        "law_status": "current" if is_current else "previous/other",
    }


def read_pages(reader):
    """full_text + offsets mapping char position -> page number."""
    parts, offsets, pos = [], [], 0
    for i, page in enumerate(reader.pages, start=1):
        t = " ".join((page.extract_text() or "").split())
        offsets.append((pos, i))
        parts.append(t)
        pos += len(t) + 1
    return " ".join(parts), offsets


def page_at(offset, offsets):
    page = 1
    for start, pg in offsets:
        if offset >= start:
            page = pg
        else:
            break
    return page


def load_chunks():
    records = []
    files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found in {PDF_DIR}/ — add your renamed PDFs and re-run.")

    step = CHUNK_SIZE - OVERLAP
    for path in files:
        meta = parse_filename(path)
        full, offsets = read_pages(PdfReader(path))

        # detect section headers once; we'll label chunks by the nearest preceding one
        marks = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(full)]
        starts = [s for s, _ in marks]
        refs = [r for _, r in marks]

        i = 0
        while i < len(full):
            piece = full[i:i + CHUNK_SIZE]
            if piece.strip():
                idx = bisect.bisect_right(starts, i) - 1  # last header at/before this chunk
                sec = refs[idx] if idx >= 0 else ""
                rec = dict(meta)
                rec["section"] = f"Sec {sec}" if sec else ""
                rec["page"] = page_at(i, offsets)
                rec["text"] = piece
                rec["chunk_id"] = f"{meta['source_file']}_{len(records)}"
                records.append(rec)
            i += step
        print(f"  {meta['source_file']}: {len(marks)} section labels detected")
    print(f"Loaded {len(files)} PDFs -> {len(records)} chunks")
    return records


def main():
    records = load_chunks()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Embedding with {EMB_MODEL} on device: {device}")
    embedder = SentenceTransformer(EMB_MODEL, device=device)
    embeddings = embedder.encode(
        [r["text"] for r in records], show_progress_bar=True, batch_size=64
    ).tolist()

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma.delete_collection(COLLECTION)
    except Exception:
        pass
    col = chroma.create_collection(COLLECTION)

    ids = [r["chunk_id"] for r in records]
    docs = [r["text"] for r in records]
    metas = [{k: r[k] for k in ("source_file", "doc_type", "year", "section", "law_status", "page")}
             for r in records]

    BATCH = 5000
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i:i + BATCH], embeddings=embeddings[i:i + BATCH],
                documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH])
        print(f"  added {min(i + BATCH, len(ids))}/{len(ids)}")
    print(f"Done. Indexed {col.count()} chunks into {CHROMA_DIR}/")


if __name__ == "__main__":
    main()