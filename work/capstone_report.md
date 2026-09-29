# Capstone Report — <your lane>

- **Author:** Reezcon Vivo
- **Lane:** Machine Learning
- **Repo:** https://github.com/reezcon/First-ML-Pipeline.git
- **Date:** September 2026

## 1. Problem framing

Managing a website with 30,000 pieces of content, or more, is overwhelming. Every day, some pieces get traffic while others get none. Some readers engage deeply (scroll, click, share), while others do not get interactions. With a limited editorial team who can improve content each week, constantly updating and editing becomes a struggle. The question is simple but time sensitive: **Which 50 pieces should we focus on?**

Without data, you might guess. You might refresh the oldest content, or the pages you remember writing, or the ones with the fanciest headlines. However, guessing wastes time, which is a critical and finite resource. You might spend a week refreshing a page that no one visits. Or you might miss a high-traffic page that's slowly losing readers. This work solves that problem. The system **ranks all 30,000 pieces by how much they'll benefit from editorial action**. The ranking uses signals we can measure, such as how many people see the page, how many click it, how long it's been since we updated it, and how long the article is. By learning which combinations of signals predict engagement, it can tell the editorial team which 50 pieces have the most potential.

Unit of analysis:
Output: 
Action:
Cost of wrong a wrong call:
Why ML helps: 

## 2. Data safety

Which data you used and which columns you deliberately excluded (and why). Leakage risks you
considered — especially label-derived fields (`trend_direction`, `trend_pct`) and pseudonymous
IDs (grouping only, never features). Confirm nothing client-identifying appears anywhere in
`work/`.

## 3. Baseline

The transparent rule or score you built first. Why it's a fair comparison, and its numbers on
the same data and metric as your model.

## 4. Model / analysis

Your method and why it fits the lane. The exact feature list (and what you left out on
purpose). The target or proxy definition, in one sentence.

## 5. Evaluation

Your split (grouped by client? time-aware?) and why. Metrics, model vs baseline **on the same
split**. What the errors look like — a short error analysis beats a big metric table.

## 6. Interpretation

What the model/clusters actually found. Feature importances or cluster profiles in plain
words. Surprises and negative results — a well-understood "no effect" is a valid result.

## 7. Recommendation

The ranked actions or decisions your output supports, and how a FlyRank editor would use them
tomorrow. State your confidence and the limits explicitly.

## 8. Reproducibility

The exact commands to re-run everything from a fresh clone, your random seeds, and your
environment (`pip freeze` highlights or `requirements.txt` deltas).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
