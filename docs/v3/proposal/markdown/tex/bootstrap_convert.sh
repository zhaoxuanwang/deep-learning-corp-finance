#!/usr/bin/env bash
# Regenerate chapters/*.md from the standalone-tex sources (../parts/*.tex).
#
# The standalone tex at the proposal root is the master; this rebuilds the
# Markdown view of it (and overwrites chapters/*.md). Run it after editing the
# tex if you want the Markdown build to catch up. Do not hand-edit both the tex
# and the Markdown, or they will diverge.
#
# Usage (from the markdown/ folder):  bash tex/bootstrap_convert.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# parts/ lives at the proposal root, one level above markdown/
PARTS=../parts

convert () {  # $1=part.tex  $2=out.md  $3="Chapter Title"  $4=chap-id
  { printf '# %s {#sec:%s}\n\n' "$3" "$4"
    # disable simple/multiline table styles so pandoc emits clean, editable pipe
    # tables; grid tables stay enabled as the fallback for the few complex tables
    # (multi-paragraph cells) that pipe cannot represent. All render to the same
    # longtable in the PDF.
    pandoc -f latex \
      -t 'markdown-auto_identifiers-simple_tables-multiline_tables' \
      --wrap=preserve --shift-heading-level-by=1 "$PARTS/$1" \
      | python3 tex/clean_md.py
  } > "chapters/$2"
}

convert commercial.tex  10_commercial.md      "Commercial Product"                          ch-commercial
convert development.tex 20_development.md      "Development Methodology and Controls"        ch-development
convert model.tex       30_model.md           "Model and Estimation"                        ch-model
convert validation.tex  40_validation.md      "Validation and Tests"                        ch-validation
convert regulatory.tex  90_appA_regulatory.md "Commercial Plan: Regulatory Compliance Map"  appA-regulatory
convert review.tex      91_appB_review.md     "Development Guidance"                         appB-guidance
convert interview.tex   92_appC_interview.md  "Interview Questions and Answers"             appC-interview

# ---- appA: the \appendix switch travels with the first appendix file -------
printf '```{=latex}\n\\appendix\n```\n\n%s' "$(cat chapters/90_appA_regulatory.md)" > chapters/90_appA_regulatory.md

# ---- appC: restore the \imgplaceholder pandoc drops (unknown command) ------
python3 - <<'PY'
import re
f = 'chapters/92_appC_interview.md'
t = open(f).read()
anchor = 'A neural net would behaves the same way.'
ph = '\n\n`\\imgplaceholder{lq\\_value\\_function.png}`{=latex}'
if 'imgplaceholder' not in t and anchor in t:
    t = t.replace(anchor, anchor + ph, 1)
    open(f, 'w').write(t)
PY

# ---- validation: re-author the float region (3 figures + 2 tables) ---------
# pandoc renders these as raw HTML / collapses the two tables to one caption,
# so the region from the fig:v3slices figure onward is replaced with clean
# pandoc-crossref Markdown. Everything above it (prose, the section heading and
# its intro paragraph) is kept verbatim.
sed -i '' '/^<figure id="fig:v3slices"/,$d' chapters/40_validation.md
# drop any trailing blank lines left behind, then append the rebuilt region
printf '%s\n' "$(cat chapters/40_validation.md)" > chapters/40_validation.md
cat >> chapters/40_validation.md <<'EOF'

::: {#fig:v3slices}
![](figures/fig_v3_slices_k.png){width=100%}

![](figures/fig_v3_slices_b.png){width=100%}

![](figures/fig_v3_slices_z.png){width=100%}

Value and policy function slices: refined network versus VFI benchmark.
:::

*Notes:* Each row varies one state variable (capital $k$, debt $b$, productivity $z$) with the others held at reference values; columns are the value $V$, investment rate $i$, gross debt $b'$, and cash $c'$. The refined network (used downstream) overlays the grid VFI benchmark; the raw network is shown for comparison.

## Monte Carlo recovery results {#sec:monte-carlo-recovery-results}

Figures[@fig:v1moments] and[@fig:v2params], with Tables[@tbl:moments] and[@tbl:params], report the Monte Carlo recovery test (Section[@sec:oracle-test-1-monte-carlo-recovery-of-parameters-and-moments]): data are simulated from known parameters, re-estimated through the full pipeline, and compared to the truth.

![Moment recovery: fitted versus true moments across replications.](figures/fig_v1_moments.png){#fig:v1moments width=90%}

*Notes:* Each panel plots the fitted (model-implied) value of one of the 11 target moments against its true value across Monte Carlo replications; the dashed line is perfect recovery ($45^\circ$) and $R^2$ in each panel title is the squared correlation. Per-moment values are in Table[@tbl:moments].

![Parameter recovery: estimated versus true parameters across replications.](figures/fig_v2_params.png){#fig:v2params width=100%}

*Notes:* Each panel plots the estimate of one of the eight structural parameters against its true value, with 95% confidence intervals; the dashed line is perfect recovery and $R^2$ in each panel title is the squared correlation. Per-parameter values are in Table[@tbl:params].

| Moment | True | Fitted | 95% CI | $R^2$ |
|:--|--:|--:|:-:|--:|
| Mean inv rate | $-0.051$ | $-0.056$ | $[-0.057,\,-0.056]$ | 0.85 |
| SD inv rate | 0.111 | 0.110 | $[0.109,\,0.111]$ | 0.64 |
| Mean op income | 0.131 | 0.137 | $[0.136,\,0.137]$ | 0.71 |
| SD op income | 0.041 | 0.047 | $[0.047,\,0.048]$ | 0.71 |
| Autocorr income | 0.251 | 0.567 | $[0.558,\,0.577]$ | 0.00 |
| Mean debt | 0.454 | 0.512 | $[0.510,\,0.514]$ | 0.71 |
| SD debt | 0.202 | 0.202 | $[0.201,\,0.203]$ | 0.36 |
| Mean cash | 0.183 | 0.195 | $[0.194,\,0.196]$ | 0.63 |
| SD cash | 0.123 | 0.117 | $[0.117,\,0.118]$ | 0.42 |
| Cash\~net debt | 0.270 | 0.293 | $[0.290,\,0.296]$ | 0.51 |
| Cash\~income | 0.514 | 0.231 | $[0.202,\,0.260]$ | 0.01 |

: Moment recovery summary. {#tbl:moments}

*Notes:* Mean across Monte Carlo replications of each of the 11 target moments: true value, fitted (model-implied) value, 95% confidence interval (firm-clustered standard errors), and $R^2$ (squared correlation between fitted and true). Corresponds to Figure[@fig:v1moments].

| Parameter | True | Estimate | RMSE | 95% CI | $R^2$ |
|:--|--:|--:|--:|:-:|--:|
| $\theta$ (returns to scale) | 0.703 | 0.703 | 0.062 | $[0.610,\,0.796]$ | 0.35 |
| $\rho$ (persistence) | 0.650 | 0.693 | 0.127 | $[0.547,\,0.839]$ | 0.03 |
| $\sigma$ (shock SD) | 0.125 | 0.135 | 0.035 | $[0.097,\,0.173]$ | 0.48 |
| $\delta$ (depreciation) | 0.111 | 0.113 | 0.032 | $[0.076,\,0.151]$ | 0.53 |
| $\gamma_1$ (convex adj. cost) | 0.525 | 0.636 | 0.330 | $[0.278,\,0.994]$ | 0.14 |
| $\gamma_0$ (fixed adj. cost) | 0.104 | 0.123 | 0.079 | $[0.023,\,0.223]$ | 0.07 |
| $\chi$ (recovery rate) | 0.454 | 0.532 | 0.252 | $[0.203,\,0.860]$ | 0.35 |
| $c_f$ (fixed op. cost) | 0.157 | 0.153 | 0.052 | $[0.087,\,0.220]$ | 0.42 |

: Parameter recovery summary. {#tbl:params}

*Notes:* Across Monte Carlo replications for each of the eight structural parameters: true value, mean estimate, root-mean-squared error, 95% confidence interval (across-fold standard deviation), and $R^2$ (squared correlation between estimate and true). Corresponds to Figure[@fig:v2params].
EOF

# ---- user manual: a hand-written placeholder chapter (no LaTeX source) -----
cat > chapters/50_user_manual.md <<'EOF'
# User Manual and Examples {#sec:ch-user-manual}

*This chapter is a placeholder for the first draft and will be completed in a later pass.*

## How to use the current code (notebook demo) {#sec:user-manual-notebook}

*[Placeholder. To be completed.]*

## How to set or tune configs {#sec:user-manual-configs}

*[Placeholder. To be completed.]*

## Hardware and software {#sec:user-manual-hardware}

*[Placeholder. To be completed.]*
EOF

echo "bootstrap conversion complete."
