import re
from rag import _provision_chunks, _query_sections, _docs

print("query sections:", _query_sections("deduction under section 80C"))
res = _provision_chunks("80C")
print("provision chunks found:", len(res))
for h in res[:6]:
    print("  p." + str(h["page"]), "|", h["text"][:120])

# every chunk whose text has a "80C." header, with the chars right after it
pat = re.compile(r"(?<![A-Za-z0-9])80C\.\s+[A-Z]")
hits = [(i, d) for i, d in enumerate(_docs) if pat.search(d)]
print("\nchunks with '80C.'+Capital:", len(hits))
for i, d in hits[:6]:
    m = pat.search(d)
    print(repr(d[m.start():m.start() + 170]))