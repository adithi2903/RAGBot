from rag import retrieve

QS = [
    "What deduction is allowed for health insurance premium?",        # 80D
    "What additional deduction is available for NPS contributions?",  # 80CCD
    "What is the deduction for interest on a home loan?",             # 24
    "By when must an individual file their income tax return?",       # 139
    "Is it mandatory to have a PAN?",                                 # 139A
    "What is the fee for late filing of an income tax return?",       # 234F
    "When is advance tax payable?",                                   # 208
]

for q in QS:
    print("\nQ:", q)
    for h in retrieve(q):
        prev = h["text"][:55].replace("\n", " ")
        print(f"   {h['source_file']:18} {str(h.get('section')):10} p.{h['page']:<4} | {prev}")
