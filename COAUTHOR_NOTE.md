# Draft note to Arthur (and re: the third-collaborator question)

*Drafted for Aadi to review, edit, and send personally. Not sent by anyone
else. Context: the paper was repaired per the integrity memo
(`cs231n-integrity-memo.md`) and rerun on leakage-free splits; it is being
prepared for arXiv + a NeurIPS 2026 workshop submission.*

---

Hey Arthur,

I want to get your sign-off on a few things before we post the CS231N paper
anywhere public.

**What changed since the course version.** After the quarter I audited the
paper the way a hostile reviewer would, and found three real problems: the
train/val split leaked task goals (the same goal strings appear on both
sides — up to 85% of val steps at the largest training size), the headline
p-values pooled the three seeds' predictions as if they were independent
(measurably anti-conservative), and the low-data claim overstated what
Phase 7 actually showed. I re-ran the statistics with an episode-clustered
analysis, rebuilt the data pipeline with goal-disjoint frozen splits plus a
genuinely untouched test set, retrained all variants on it, and rewrote the
paper around "when does action-type supervision help, and which
implementation/evaluation choices fake it" — the M-RoPE bug, the leakage
diagnosis, and the stats repair are now first-class contributions instead
of footnotes. The honest headline is weaker than the course version but it
holds up. Repo has everything; `REVIEW.md` summarizes it.

**What I need from you:**
1. **Author list + order.** Current draft: me and you, in that order, no
   equal-contribution marker (course version had asterisks — happy to
   discuss restoring them; your call matters here).
2. **Contribution statement.** Draft says: you — Stage-1 classifier study,
   data pipeline and error analysis, figures, editing; me —
   infrastructure, Stage-2 variants, experiment orchestration, statistical
   reanalysis and confirmatory protocol, writing. Please edit until it's
   factually right by your accounting; nothing goes out until you approve
   the exact wording.
3. **AI disclosure.** The paper discloses extensive LLM-assistant use for
   code, orchestration, the audit, and drafting, with us owning design and
   verification. Confirm you're comfortable with the wording.
4. **Consent to venues.** Plan: arXiv preprint (cs.CV, cross-list cs.AI)
   + submission to "Evaluation of Interactive Agents" @ NeurIPS 2026
   (non-archival, deadline Aug 29 AoE). Both need your explicit OK.

**Third-collaborator question.** The original project proposal listed a
third collaborator. The final experiments, code, and manuscript came from
the two of us as far as I can reconstruct, but I don't want to make that
call unilaterally: do you remember any contribution from them (ideas,
code, data work, writing) that rises to authorship or acknowledgment
level? If there's any doubt, I'd rather reach out to them directly before
we post. Flag anything you remember.

No rush on a same-day reply, but the workshop deadline is Aug 29, so I'd
like to close authorship/consent well before then.

— Aadi
