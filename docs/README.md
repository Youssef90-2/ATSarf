# docs

## `FYP_Report.html` — the project report

~12,800 words, 19 chapters/appendices, 38 tables, 11 figures. Formatted for A4
with page breaks before each chapter, so it paginates sensibly in Word.

### Opening it in Word

1. Word → **File → Open** → select `FYP_Report.html` (set the file-type filter to
   *All Files* if it isn't shown).
2. Word renders it with headings, tables and page breaks intact.
3. **File → Save As → Word Document (.docx)**.

Do **not** copy-paste from a browser — that loses the table borders and the page
breaks. Opening the file directly is what preserves them.

### After converting

Two things are worth doing once in Word:

- **Insert a real table of contents.** The static one in the file has no page
  numbers. Delete it, then References → Table of Contents → Automatic; the
  headings are already `Heading 1/2/3`, so it will populate correctly.
- **Check the Arabic.** The CSS asks for *Traditional Arabic*; if your Word
  substitutes something that renders poorly, select all and set the complex-script
  font to *Traditional Arabic*, *Amiri* or *Sakkal Majalla*.

### What still needs filling

Chapter 10 (Results) is written but its tables are empty by design — the gold
annotations do not exist yet. Section 10.1 gives the annotation protocol, 10.2 the
five experiments, and 10.3 the tables ready to receive numbers. The original
paper's figures sit in the right-hand column of each for orientation.

Everything else is complete and does not depend on the annotation.

### Regenerating

The report is hand-written prose, not generated from the code, so there is no build
step. If a measurement changes, the places to update are:

| Where | What |
|---|---|
| §5.5 | honorific counts (1,031 of 8,986) |
| §6.8, Table 6.2 | graph construction before/after |
| §7.7, Table 7.2 | the two-condition structural comparison |
| §9.4, Table 9.4 | test counts |
| Appendix A | module line counts |
| Appendix C | invariant status |

`PORT_NOTES.md` in the parent directory is the technical companion: it records every
change made during the port with its source-line justification, and is the place to
look when a claim in the report needs backing.
