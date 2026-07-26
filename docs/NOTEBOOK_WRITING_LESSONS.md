# Notebook Writing Lessons

Written after a long LaTeX report editing session that surfaced repeated, serious writing/verification failures. The report is not the point here — the same failure modes apply directly to writing markdown cells and code comments in the project's Jupyter notebooks, which is the actual target this document is for.

## Problems found this session

1. **AI-narrator scaffolding.** Sentences that explain their own presence or point forward instead of stating a fact: "This is the approximation of X introduced in Section Y", "we will see that...", "as described next". Confirmed present in report prose that had supposedly been cleaned of this once already, added back in during a later editing pass. Reads as a chatbot narrating its own structure to the reader instead of a document stating facts.

2. **Fabricated technical content presented with confidence.** An equation (a loss function) was written from pattern-completion/memory, not checked against the cited paper, the actual library implementation, or any other findable source. It matched nothing anywhere. This is the most serious failure category: presenting an unverified claim as if it were verified.

3. **Close paraphrase without quotation.** Reusing 4-8+ word phrases from a source almost verbatim, with only trivial synonym swaps, while attaching a citation as if the citation alone licenses the wording. A citation permits attribution. It does not license verbatim or near-verbatim reuse — that still needs quotation marks, or a genuinely independent rewording.

4. **Using an abbreviation before it's ever defined.** VI, LA, MOPED, NLL, RMSE, ECE, ID, OOD, CDF, DNN, MC were all used bare in running text well before their first expansion, sometimes 100+ lines earlier. The rule is: full term (ABBREV) at the true first use, bare abbreviation after that, never the reverse.

5. **Referencing something the reader hasn't been introduced to yet.** A conceptual/theory section named specific experiments before any section had told the reader those experiments exist. This only makes sense to a writer holding the entire document in context at once — a linear human reader hits it cold. Check what the reader actually knows at that exact point in a start-to-end read, not what the writer knows about the whole document.

6. **Stale build artifacts.** A figure was regenerated in one location but never synced to the separate directory that actually gets compiled/read, and the fix was reported as done based on the wrong copy. When a fix touches a generated asset, check every location that consumes it, not just the one that produced it.

7. **Taking action instead of answering.** A continuation cue ("so?") or an instruction about response style ("batch your answers") was misread as authorization to run further commands (recompiles, file writes) instead of just answering the question that was actually asked. If the user asks a question, answer it and stop. Act only on an explicit instruction to act.

8. **Misdiagnosing what feedback actually referred to.** Assumed a complaint pointed at the most recently discussed bug instead of checking what was literally said. Assumed "bad references" meant citation-key/cross-reference validity when it meant plagiarism. When feedback is even slightly ambiguous, ask which specific thing it means rather than picking the interpretation that's easiest to act on immediately.

9. **Using a tool that doesn't fit the job and nearly trusting its output anyway.** A code-plagiarism detector was run against prose, returned an all-zero result, and that null result almost got reported as meaningful — even though it directly contradicted matches already confirmed by hand. A tool returning a suspiciously clean result needs a sanity check against a known-true case before it's trusted.

## What actually got fixed, once caught

- Every abbreviation moved to its true first use, expanded once, used bare after.
- Dataset names removed from the conceptual/theory content and confined to the sections that actually introduce those experiments.
- Three instances of close-paraphrase-without-quotation reworded or given proper attribution.
- A fabricated equation replaced with the formula actually implemented in the library code being described, verified by reading that code directly.
- Two misdirected cross-references (pointing at a section that didn't contain what the sentence claimed) corrected after checking what the target section actually says.
- A block of broken comment syntax that was silently rendering as visible garbage text in a compiled document, removed.
- All of the above independently re-checked by fresh audit passes rather than self-graded, specifically because self-grading had already missed real issues once.

## What the standing rules are, going forward

- No em dashes or double dashes anywhere — prose, code comments, markdown cells.
- No self-referential narration. State facts. Don't narrate the document's own structure, don't explain why a sentence is being said, don't reassure the reader that something will be covered later unless it's a genuine, single, top-level roadmap statement.
- Every abbreviation gets "Full Term (ABBREV)" once, at its real first use — not before, not redefined again later.
- Never name something the reader hasn't been introduced to yet, in the order they'll actually read it.
- Any claim drawn from an external source: either paraphrase it in genuinely different words, or quote it exactly with quotation marks and a citation. Never both a citation and near-verbatim wording — that's still plagiarism.
- Never write a formula, an API detail, or a factual claim from memory when the source is checkable. Fetch the paper. Read the actual code. If it can't be checked, say so plainly instead of writing it with false confidence.
- When a fix touches a generated file (figure, output, checkpoint), check every place that consumes a copy of it, not just the place that generated it.
- Don't infer permission to act from an ambiguous continuation. Answer what was asked. Act only when told to.
- When corrected, re-derive the claim from the actual current file/state before restating it — don't assume the prior explanation was right just because it was already given once.

## Audits run this session, for reference

- Abbreviation-ordering sweep + table-count verification (independent agent, read-only).
- Dataset-name-leakage + citation-key validity + cross-reference-target validity + section-by-section prose logic pass (independent agent, read-only).
- AI-narrator-scaffolding sweep, specifically targeting the pattern in item 1 above (independent agent, read-only).
- Source-comparison plagiarism audit against every literature PDF actually cited (independent agent, read-only, followed by a manual word-level n-gram overlap check as a second, tool-based cross-check once the first attempted tool turned out to be code-oriented and unusable on prose).
- Direct verification of individual claims against primary sources fetched live (the actual Flipout paper, Laplace Redux, the PML textbook, the Krishnan AvUC paper, the Kuleshov calibration paper, the Osawa tau-schedule paper) and against the actual installed library source code (`bayesian_torch.utils.avuc_loss`), rather than trusting a citation key to mean the nearby claim was already checked.

## Applying this to notebook markdown and code comments specifically

- A markdown cell explaining what the next code cell does should state what it does and, if non-obvious, why a specific choice was made — not narrate "now we will..." unless it's the notebook's actual top-of-file roadmap.
- If a cell implements something from a paper or a library's own method, check the description against the actual paper or the actual installed source before writing it into a markdown cell or docstring. Do not describe a formula, a hyperparameter schedule, or an API's behavior from memory.
- Don't reference a variant, dataset, or result in an early overview cell before the notebook has actually defined or loaded it.
- Code comments stay to one line, stating a non-obvious constraint or reason — never multi-line narration of what the code already makes obvious.
- The same plagiarism standard applies to notebook prose as to the report: reusing a paper's or a library README's phrasing to explain a cell needs either a genuine rewrite or a quoted, attributed excerpt, not a close paraphrase with a bare citation.
