# Indian Tax Law RAG Chatbot

A chatbot that answers questions about Indian income-tax law. Instead of letting an
LLM make things up, it retrieves the relevant passages from the actual Income-tax Act
and answers **only from those sources, with citations** (document, section, page).

Built as an internship project to learn how to build a real, grounded RAG system.

## What it does
- Answers income-tax questions using the **Income-tax Act, 1961** and **Income-tax Rules, 1962**
- Cites the exact document, section, and page for every answer
- Shows a ✅ / ⚠️ badge that verifies the cited sections are actually in the sources
- Says "I could not find this" instead of guessing when the answer isn't in the documents

## How it works
1. The PDFs are split into chunks and stored in a vector database (Chroma).
2. For each question it does **hybrid retrieval** — keyword search (BM25) + meaning search
   (embeddings), combined and then re-ranked — to find the most relevant passages.
3. If the question names a section (e.g. 80C), it specifically finds where that provision
   begins, not just where it's mentioned.
4. **NVIDIA Nemotron** writes the answer using only those passages and cites them.
5. An audit step checks every cited section really appears in the retrieved text.

## Tech used
NVIDIA Nemotron (LLM) · sentence-transformers (embeddings) · BM25 (keyword search) ·
cross-encoder (re-ranking) · Chroma (vector DB) · pypdf · Gradio (chat UI)

## Results
Evaluated on a 20-question test set with known correct sections:
- Retrieval hit-rate: ~70%
- Citation accuracy: ~80% (up from a 45% baseline as I improved retrieval)

Works best on common deductions and section-named questions; some procedural sections
(filing / PAN / penalties) are weaker — documented honestly rather than hidden.

## What I learned
Most of the work was not wiring the pieces together but **measuring** the system and
fixing where retrieval actually broke — cross-references drowning the right provision,
the table of contents hijacking results, and two laws with conflicting section numbers.
Building an evaluation harness is what made those fixes possible.

*Educational project — not professional tax advice.*
