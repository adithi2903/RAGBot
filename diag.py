import chromadb
from config import CHROMA_DIR, COLLECTION
from rag import retrieve

# 1) Do chunks with real 80C text even exist?
col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
alld = col.get(include=["documents", "metadatas"])
have = [(m, d) for d, m in zip(alld["documents"], alld["metadatas"]) if "80C" in d]
print(f"Chunks containing '80C': {len(have)}\n")

# 2) Look at what retrieval actually returns (the full text, not just labels)
print("=== Retrieved for 'deduction under section 80C' ===")
for h in retrieve("deduction under section 80C"):
    print(f"\n--- {h['source_file']} {h.get('section')} p.{h['page']} ---")
    print(h["text"][:280])