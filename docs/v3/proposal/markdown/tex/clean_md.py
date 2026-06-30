#!/usr/bin/env python3
"""Post-process pandoc latex->markdown output into the chapter source format.

Run after:  pandoc -f latex -t markdown --wrap=preserve --shift-heading-level-by=1 part.tex

Three mechanical fixes (see the conversion notes in the proposal README):
  1. Collapse the empty fenced div that pandoc wraps around sub/subsub headings
     (`::: {#slug}` / heading / `:::`) back into an inline `## Heading {#slug}`.
  2. Prefix every heading id with `sec:` (pandoc-crossref needs it to treat the
     heading as a numbered, referenceable section). fig:/tbl:/sec: are left alone.
  3. Rewrite the verbose pandoc cross-reference
        [..](#anchor){reference-type="ref" reference="SLUG"}
     to a crossref `[@sec:SLUG]` / `[@fig:..]` / `[@tbl:..]`, consuming one
     preceding space/nbsp so crossref's own nonbreaking space is not doubled.
     Table labels `tab:` are renamed to crossref's `tbl:`.

Usage:  clean_md.py < in.md > out.md
"""
import re
import sys

text = sys.stdin.read()

# --- 1. collapse heading-id divs -------------------------------------------
# pandoc wraps sub/subsub headings in an empty fenced div carrying the id:
#   ::: {#slug}
#   ## Heading text            (may already carry its own inline {#slug})
#   :::
# Drop the wrapper, keeping the heading. If the heading has no inline id, add
# the div's id; if it already has one (auto_identifiers disabled), leave it.
def _collapse(m):
    slug, head = m.group(1), m.group(2)
    if re.search(r'\{#[^}]+\}\s*$', head):
        return head
    return f'{head} {{#{slug}}}'

text = re.sub(
    r'^::: \{#([^}]+)\}\n(#{1,6} [^\n]*?)\n:::$',
    _collapse,
    text,
    flags=re.MULTILINE,
)

# --- 2. prefix heading ids with sec: ---------------------------------------
def _prefix_heading(m):
    head, slug = m.group(1), m.group(2)
    if re.match(r'(sec|fig|tbl|eq):', slug):
        return m.group(0)
    return f'{head}{{#sec:{slug}}}'

text = re.sub(
    r'^(#{1,6} .*?)\{#([^}]+)\}$',
    _prefix_heading,
    text,
    flags=re.MULTILINE,
)

# --- 3. rewrite verbose refs (consume one preceding space/nbsp) -------------
def _ref(m):
    slug = m.group('slug')
    if slug.startswith('fig:'):
        tag = '@' + slug
    elif slug.startswith('tab:'):
        tag = '@tbl:' + slug[len('tab:'):]
    elif slug.startswith(('sec:', 'tbl:', 'eq:')):
        tag = '@' + slug
    else:
        tag = '@sec:' + slug
    return f'[{tag}]'

# optional single preceding whitespace (regular or nbsp  ), then the link
ref_pattern = re.compile(
    r'[  ]?'
    r'\[[^\n]*?\]\(#[^)]*\)'
    r'\{reference-type="[^"]*"\s+reference="(?P<slug>[^"]+)"\}'
)
text = ref_pattern.sub(_ref, text)

sys.stdout.write(text)
