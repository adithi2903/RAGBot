"""
The RAG core:
  HYBRID retrieval (dense embeddings + BM25 keyword, fused with RRF)
  + SECTION-AWARE boost (if the question names a section, force in the chunk where
    that provision actually begins)
  -> cross-encoder rerank -> Nemotron answers using ONLY those chunks, with citations.
"""
import os
# Use locally-cached models; don't ping Hugging Face every run (it can hang).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

from config import (client, MODEL, CHROMA_DIR, COLLECTION,
                    EMB_MODEL, RERANK_MODEL, TOP_K, POOL)

# --- load models + saved index once ---
_embedder = SentenceTransformer(EMB_MODEL)
_reranker = CrossEncoder(RERANK_MODEL)
_col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)

_all = _col.get(include=["documents", "metadatas"])
_ids, _docs, _metas = _all["ids"], _all["documents"], _all["metadatas"]


def _tok(s):
    return re.findall(r"\w+", s.lower())


_bm25 = BM25Okapi([_tok(d) for d in _docs])

# Flag "table of contents / arrangement of sections" chunks: they cram many
# section headers together and otherwise hijack retrieval. Real provisions have
# one header then lots of text.
_id2idx = {c: i for i, c in enumerate(_ids)}
_HEADER_PAT = re.compile(r"(?<![A-Za-z0-9])\d{1,3}[A-Z]{0,3}\.\s+[A-Z]")


def _looks_like_toc(d):
    # A table-of-contents chunk has MANY headers packed TIGHTLY (just titles).
    # A real provision region may have several headers but with lots of text between.
    n = len(_HEADER_PAT.findall(d))
    return n >= 4 and (len(d) / max(n, 1)) < 80


_toc_mask = [_looks_like_toc(d) for d in _docs]
_n_toc = sum(_toc_mask)


def _is_toc(cid):
    return _toc_mask[_id2idx[cid]]


print(f"[rag] hybrid index ready: {len(_docs)} chunks "
      f"(dense + BM25 + section-aware; {_n_toc} TOC chunks excluded)")


def _hit(idx):
    return {"id": _ids[idx], "text": _docs[idx], **_metas[idx]}


def dense_search(query, n):
    res = _col.query(query_embeddings=_embedder.encode([query]).tolist(), n_results=n)
    return [{"id": c, "text": d, **m}
            for c, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


def sparse_search(query, n):
    scores = _bm25.get_scores(_tok(query))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [_hit(i) for i in top]


def rrf(result_lists, k=60):
    score, item = {}, {}
    for lst in result_lists:
        for rank, h in enumerate(lst):
            score[h["id"]] = score.get(h["id"], 0.0) + 1.0 / (k + rank + 1)
            item[h["id"]] = h
    return [item[i] for i in sorted(score, key=lambda x: -score[x])]


def _query_sections(q):
    """Pull section refs from the question: 'section 80C', '80C', '115BAC'."""
    secs = re.findall(r"sections?\s+(\d{1,3}[A-Za-z]{0,3})", q, flags=re.I)
    secs += re.findall(r"\b(\d{1,3}[A-Za-z]{1,3})\b", q)   # numbers WITH letters (80C, 115BAC)
    return list({s.upper() for s in secs})


def _provision_chunks(sec, limit=8):
    """Find chunks where this provision actually BEGINS (e.g. '80C. Deduction ...').

    Key test: a real provision header is followed by a long BODY, whereas a
    table-of-contents entry is immediately followed by the NEXT section header.
    So we reject any match that has another header within 200 chars after it.
    """
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(sec)}\.\s+[A-Z]")
    out = []
    for i, d in enumerate(_docs):
        m = pat.search(d)
        if not m:
            continue
        tail = d[m.end(): m.end() + 200]
        if _HEADER_PAT.search(tail):     # another header right after -> TOC/list entry
            continue
        h = _hit(i)
        h["section"] = f"Sec {sec}"      # we KNOW the section here -> correct citation
        out.append(h)
        if len(out) >= limit:
            break
    return out


def _rerank(question, hits):
    if not hits:
        return []
    scores = _reranker.predict([(question, h["text"]) for h in hits])
    return [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]


_EXPAND_SYS = ("detailed thinking off\n"
               "Rewrite the user's Indian income-tax question into a short search query using the "
               "FORMAL statutory wording the Income-tax Act would use. Expand common terms, e.g. "
               "'home loan'->'interest on borrowed capital house property'; 'PAN'->'permanent account number'; "
               "'late filing fee'->'fee default furnishing return of income'; 'NPS'->'pension scheme central government'; "
               "'health insurance'->'health insurance premia mediclaim'. "
               "Output ONLY the expanded keywords, no explanation, no tags.")


def _expand_query(q):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": _EXPAND_SYS},
                      {"role": "user", "content": q}],
            temperature=0.0, max_tokens=60,
        )
        out = (r.choices[0].message.content or "").strip()
        return out.splitlines()[0][:200] if out else ""
    except Exception:
        return ""


def retrieve(question, k=TOP_K, pool=POOL, expand=False):
    # query expansion costs an extra model call; off by default for a fast UI,
    # can be turned on (e.g. in evaluation) for a small recall boost.
    search_q = question
    if expand:
        e = _expand_query(question)
        search_q = (question + " " + e).strip()
    fused = rrf([dense_search(search_q, pool), sparse_search(search_q, pool)])

    # section-aware: collect the chunks where named provisions BEGIN
    forced, seen = [], set()
    for sec in _query_sections(search_q):
        for h in _provision_chunks(sec):
            if h["id"] not in seen:
                seen.add(h["id"])
                forced.append(h)

    # Rank the two groups separately, then GUARANTEE the provision chunks lead.
    # Drop table-of-contents/list chunks from the fallback results too.
    forced = _rerank(question, forced)[:k]
    rest = _rerank(question, [h for h in fused if h["id"] not in seen and not _is_toc(h["id"])])
    return (forced + rest)[:k]


def _format_sources(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        tag = f"[{i}] {h['source_file']}"
        if h.get("section"):
            tag += f", {h['section']}"
        tag += f", p.{h['page']}"
        blocks.append(f"{tag}\n{h['text']}")
    return "\n\n".join(blocks)


# Nemotron turns off its long internal reasoning ONLY when the system message is
# just this toggle. Task instructions therefore go in the user turn (below).
THINK = "detailed thinking off"

INSTRUCTIONS = """You are an assistant for Indian income-tax law. Answer ONLY using the SOURCES provided.
Rules:
- Be concise (3-5 sentences). Every factual statement must cite its source like (Document, Section, p.Page).
- The sources are the Income-tax Act, 1961 and the Income-tax Rules, 1962. Use the section/rule numbers exactly as they appear in the sources.
- If the answer is not in the SOURCES, say: "I could not find this in the provided documents." Never use outside knowledge or guess.
- Put ONLY the final answer between <answer> and </answer> tags. Write it directly, with no reasoning."""


def _clean(text):
    """Strip any leaked reasoning; keep only what's inside <answer>...</answer>."""
    if not text:
        return ""
    if "<answer>" in text:
        inner = text.split("<answer>", 1)[1]
        return inner.split("</answer>")[0].strip()
    # fallback: model ignored the tags -> strip leading chain-of-thought sentences
    t = text.strip()
    opener = re.compile(
        r"^(okay|alright|sure|first|second|next|now|let me|let us|let's|i need|i'll|"
        r"i will|i recall|i should|to answer|to determine|looking|checking|based on the)\b",
        re.I)
    sents = re.split(r"(?<=[.!?])\s+", t)
    while len(sents) > 1 and opener.match(sents[0].strip()):
        sents.pop(0)
    return " ".join(sents).strip() or t


def _messages(question, hits):
    return [
        {"role": "system", "content": THINK},
        {"role": "user", "content": f"{INSTRUCTIONS}\n\nSOURCES:\n{_format_sources(hits)}\n\nQUESTION: {question}"},
    ]


def audit(answer_text, hits):
    """Trust check: every section the answer CITES must actually appear in the
    retrieved sources. Returns (status, unsupported_sections).
      status 'ok'   -> all cited sections are grounded in the sources
      status 'warn' -> a cited section is NOT in the sources (possible hallucination)
      status 'none' -> the answer cited no section (e.g. a refusal)
    Pure-Python, no extra model call."""
    cited = set(re.findall(r"[Ss]ec(?:tion)?\.?\s*([0-9]{1,3}[A-Z]{0,3})", answer_text or ""))
    if not cited:
        return "none", []
    corpus = " ".join(h["text"] for h in hits) + " " + \
             " ".join((h.get("section") or "") for h in hits)
    unsupported = [c for c in cited
                   if not re.search(rf"(?<![0-9A-Za-z]){re.escape(c)}(?![0-9A-Za-z])", corpus)]
    return ("warn" if unsupported else "ok"), unsupported


def answer(question):
    hits = retrieve(question)
    if not hits:
        return "No documents indexed yet. Run `python ingest.py` first.", []
    resp = client.chat.completions.create(
        model=MODEL, messages=_messages(question, hits),
        temperature=0.2, top_p=0.9, max_tokens=1024,
    )
    msg = resp.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    text = _clean(text) or "I could not find this in the provided documents."
    return text, hits


def answer_stream(question):
    hits = retrieve(question)
    if not hits:
        yield "No documents indexed yet. Run `python ingest.py` first.", []
        return
    stream = client.chat.completions.create(
        model=MODEL, messages=_messages(question, hits),
        temperature=0.2, top_p=0.9, max_tokens=1024, stream=True,
    )
    partial = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            partial += delta
            yield partial, hits   # raw text; the UI hides reasoning until <answer> opens


if __name__ == "__main__":
    print("\n=== retrieve('deduction under section 80C') ===")
    for h in retrieve("deduction under section 80C"):
        print(f"{h['source_file']} {h.get('section')} p.{h['page']}  |  {h['text'][:70]}...")