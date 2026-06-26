import re
from rag import _docs

count = 0
for d in _docs:
    for m in re.finditer(r"80C\.", d):
        s = m.start()
        print(repr(d[max(0, s-8):s+45]))
        count += 1
        if count >= 12:
            break
    if count >= 12:
        break