# 🇮🇳 Indian Tax Law RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indian
income-tax law. It retrieves from **official tax PDFs** and answers using **NVIDIA
Nemotron**, with every answer backed by **citations** (document, section, page) so
responses are verifiable and grounded — not made up.

Built as an internship project to demonstrate a practical, citation-native RAG system.

## ✨ Features
- Answers grounded only in official source documents (no hallucinated tax advice)
- Inline **citations** with document name, section, and page number
- Distinguishes the **current law (Income-tax Act, 2025)** from the **previous law (Act, 1961)**
- Clean chat interface (Gradio)
- Says "not found in the provided documents" instead of guessing

## 🏗️ How it works
```
PDFs → chunk (with metadata) → embeddings → vector DB (Chroma)
                                                     │
question → embed → retrieve top chunks → (rerank) → Nemotron answers + cites sources
```

## 🧰 Tech stack
| Part | Tool |
|------|------|
| LLM (answers) | NVIDIA Nemotron (via NVIDIA API) |
| PDF reading | pypdf |
| Embeddings | sentence-transformers |
| Vector database | Chroma |
| UI | Gradio |

## 📄 Documents used
Official sources (download separately — not committed to the repo):
- Income-tax Act, 1961 — https://www.indiacode.nic.in/handle/123456789/2435
- Income-tax Act, 2025 — https://www.incometax.gov.in/iec/foportal/newdownloads/income-tax-act-2025
- Income-tax Rules, 1962, latest Finance Act, and CBDT circulars — https://www.incometaxindia.gov.in

PDFs are named with the convention `DocType_Year_Section_Topic.pdf` (e.g.
`ITAct_1961_Sec80C_Deductions.pdf`) so the filename becomes searchable metadata.

## 🚀 Setup
```bash
pip install -r requirements.txt
export NVIDIA_API_KEY="nvapi-..."   # get a free key at https://build.nvidia.com
```
1. Download the PDFs above, rename them, and put them in a `tax_pdfs/` folder.
2. Open `Tax_RAG_Chatbot.ipynb` and run the cells top to bottom.
3. The last cell launches the chat UI.

## 📊 Results
- Tested on common income-tax queries (deductions, filing dates, definitions).
- Answers cite the exact source document, section, and page.
- _(Add your own example screenshots in `screenshots/`.)_

## 🔭 Future work
Production scaling path: hybrid retrieval (BM25 + dense), cross-encoder reranking,
a post-answer citation-audit layer, temporal/version routing, and a FastAPI +
vector-DB service for concurrent users.

## ⚠️ Disclaimer
This is an educational project and not professional tax or legal advice. Always
verify against the official source documents (which the bot links to).