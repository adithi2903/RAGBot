from rag import answer
a, hits = answer("What is the deduction available under section 80C?")
print(a)
print("\nSources:", [(h["source_file"], h["section"], h["page"]) for h in hits])