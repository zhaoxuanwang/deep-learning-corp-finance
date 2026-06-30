# Proposal

There are **two ways** to build this proposal. They share `figures/` and produce the
same document. The **standalone LaTeX** at the root is the master; the **Markdown** option
in `markdown/` is a regenerable view of it.

```
proposal/
  figures/            shared by both builds
  proposal.tex        OPTION 1: standalone LaTeX (master) -- edit + compile here
  parts/*.tex
  proposal.pdf
  markdown/           OPTION 2: Markdown -> Pandoc -> PDF
```

## Option 1 -- Standalone LaTeX (primary)

Edit the chapter bodies in `parts/*.tex` (semantic names: `commercial`, `model`,
`validation`, `regulatory`, ...) directly, then compile at the proposal root:

```
latexmk -pdf proposal.tex      # -> proposal.pdf
latexmk -c proposal.tex        # remove .aux/.log/... intermediates
```

`proposal.tex` is a thin manifest of `\partchapter{<file>}{<Title>}` lines:

- **Reorder** chapters by reordering those lines (numbers, ToC, and `\ref`
  cross-references all renumber automatically; the parts carry no hard-coded numbers).
- **Mute** a chapter by commenting its line out (`% \partchapter{...}`).

Structure (model -> solvers -> simulation & estimation): each solver lives in its
own `parts/solver_*.tex` and each estimation method in `parts/est_*.tex`, pulled in
by the chapter wrappers `parts/solvers.tex` and `parts/estimation.tex`. Add a solver
or method by dropping in a new file and adding one `\input` line there.

Migration note: `parts/model.tex` is the legacy combined chapter, parked (commented
out in the manifest) as the source to move into CH2-CH4. Until that content is moved,
`latexmk` reports undefined references (e.g. `panel-simulation`) from `validation.tex`
pointing into the parked file; they resolve once the labelled sections are migrated.

Uses pdflatex with `libertinus` + `libertinust1math`.

## Option 2 -- Markdown (lightweight alternative)

Edit the `.md` files in `markdown/chapters/`, then build from inside `markdown/`:

```
cd markdown
make pdf       # published set (Commercial, Model, Validation)  -> build/proposal.pdf
make full      # every chapter + appendices                     -> build/proposal_full.pdf
make preview CH=chapters/30_model.md                            -> build/preview.pdf
make clean
```

Uses Pandoc + pandoc-crossref + xelatex (`brew install pandoc pandoc-crossref`; needs a TeX
install with `xelatex`, `libertinus-otf`, `inconsolata`). All output stays in `markdown/build/`.
Which chapters are included is the `CHAPTERS` / `CHAPTERS_FULL` file lists in `markdown/Makefile`.

### Markdown writing conventions

- **Headings.** `#` chapter, `##` section, `###` subsection. To cross-reference a heading,
  give it a `sec:` id: `## Grid refinement {#sec:grid-refinement}`.
- **Cross-references** (pandoc-crossref). Keep the literal word in prose; the filter adds the
  number: `Section [@sec:grid-refinement]` -> "Section 2.6.1", `Figure [@fig:v1moments]`,
  `Table [@tbl:moments]`. Pairs/ranges work: `Sections [@sec:a] and [@sec:b]`.
- **Equations.** Single-line display math with the paper number:
  `$$ q = \frac{1}{1+r_f} \tag{6} $$`, referred to in prose as "Eq. 6".
- **Figures.** `![Caption.](figures/foo.png){#fig:foo width=90%}`. Several stacked images
  under one caption: a `::: {#fig:foo}` div (see `chapters/40_validation.md`).
- **Tables.** Pipe tables with a crossref caption: `: Caption. {#tbl:foo}`.
- **Citations** off by default; to use `[@bibkey]`, add entries to `references.bib` and
  uncomment `citeproc` + `bibliography` in `defaults.yaml`.

## Keeping the two in sync

`proposal.tex` / `parts/` is the master. The Markdown is a snapshot of it. After editing the
tex, regenerate the Markdown view with:

```
cd markdown && bash tex/bootstrap_convert.sh
```

This rebuilds `markdown/chapters/*.md` from `../parts/*.tex`. **Do not hand-edit both** the
tex and the Markdown for the same content, or they will drift apart.
