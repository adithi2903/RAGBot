"""
Evaluation harness for the tax RAG bot.

    python evaluate.py

Runs a fixed "golden set" of tax questions whose correct section is known, then
measures, with numbers:
  - Retrieval hit-rate : did we retrieve a chunk for the right section?
  - Citation accuracy  : did the answer cite the right section?
  - Keyword coverage   : did the answer contain the expected key facts?
  - Answer rate        : did it answer (vs refuse)?
Results print as a table and are saved to eval_results.csv for your report.
"""
import csv
import re
from rag import answer

# --- Golden set: question, expected section(s), expected key facts -------------
# Section numbers are the ground truth (from the Income-tax Act).
GOLD = [
    {"q": "What is the maximum deduction available under section 80C?",
     "sec": ["80C"], "kw": ["1,50,000", "life insurance"]},
    {"q": "What deduction is allowed for health insurance premium?",
     "sec": ["80D"], "kw": ["health", "insurance"]},
    {"q": "Is interest on an education loan deductible?",
     "sec": ["80E"], "kw": ["interest", "loan"]},
    {"q": "What deduction is available for donations to charitable institutions?",
     "sec": ["80G"], "kw": ["donation"]},
    {"q": "What is the deduction on interest from a savings bank account?",
     "sec": ["80TTA"], "kw": ["savings", "interest"]},
    {"q": "What additional deduction is available for NPS contributions?",
     "sec": ["80CCD"], "kw": ["pension", "national"]},
    {"q": "What is the rebate available under section 87A?",
     "sec": ["87A"], "kw": ["rebate"]},
    {"q": "What is the standard deduction from salary income?",
     "sec": ["16"], "kw": ["salary", "deduction"]},
    {"q": "What is the deduction for interest on a home loan?",
     "sec": ["24"], "kw": ["interest", "property"]},
    {"q": "By when must an individual file their income tax return?",
     "sec": ["139"], "kw": ["return"]},
    {"q": "Is it mandatory to have a PAN?",
     "sec": ["139A"], "kw": ["permanent account number"]},
    {"q": "What is the fee for late filing of an income tax return?",
     "sec": ["234F"], "kw": ["fee"]},
    {"q": "How is tax deducted at source on salary?",
     "sec": ["192"], "kw": ["deduct", "salary"]},
    {"q": "What is presumptive taxation for small businesses?",
     "sec": ["44AD"], "kw": ["presumptive", "business"]},
    {"q": "What is presumptive taxation for professionals?",
     "sec": ["44ADA"], "kw": ["profession"]},
    {"q": "What capital gains exemption is available on sale of a residential house?",
     "sec": ["54"], "kw": ["capital gain", "residential"]},
    {"q": "How is residential status of a person determined?",
     "sec": ["6"], "kw": ["resident"]},
    {"q": "What are the heads of income under the Income-tax Act?",
     "sec": ["14"], "kw": ["salaries", "income"]},
    {"q": "What is the new tax regime under section 115BAC?",
     "sec": ["115BAC"], "kw": ["regime", "rate"]},
    {"q": "When is advance tax payable?",
     "sec": ["208"], "kw": ["advance tax"]},
]


def retrieved_right_section(hits, secs):
    for h in hits:
        label = (h.get("section") or "").upper()
        for s in secs:
            if s.upper() in label:
                return True
            # or the provision header appears in the chunk text
            if re.search(rf"(?<![0-9A-Za-z]){re.escape(s)}\.", h["text"]):
                return True
    return False


def run():
    rows = []
    R = C = A = 0
    Kcov = 0.0
    print(f"Evaluating {len(GOLD)} questions...\n")
    for g in GOLD:
        ans, hits = answer(g["q"])
        ans = ans or ""
        ans_l = ans.lower()
        answered = "could not find" not in ans_l
        rhit = retrieved_right_section(hits, g["sec"])
        chit = any(s.lower() in ans_l for s in g["sec"])
        kcov = sum(1 for k in g["kw"] if k.lower() in ans_l) / max(len(g["kw"]), 1)

        R += rhit; C += chit; A += answered; Kcov += kcov
        rows.append({
            "question": g["q"],
            "expected_sections": "|".join(g["sec"]),
            "retrieval_hit": rhit,
            "citation_correct": chit,
            "keyword_coverage": round(kcov, 2),
            "answered": answered,
            "answer": ans.replace("\n", " ")[:500],
        })
        print(f"{'HIT ' if rhit else 'MISS'} | cite {'Y' if chit else 'N'} "
              f"| kw {kcov:.0%} | {g['q'][:55]}")

    n = len(GOLD)
    print("\n================ RESULTS ================")
    print(f"Retrieval hit-rate    : {R}/{n} = {R/n:.0%}")
    print(f"Citation accuracy     : {C}/{n} = {C/n:.0%}")
    print(f"Avg keyword coverage  : {Kcov/n:.0%}")
    print(f"Answer rate (not refused): {A}/{n} = {A/n:.0%}")

    with open("eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nSaved per-question details to eval_results.csv")


if __name__ == "__main__":
    run()