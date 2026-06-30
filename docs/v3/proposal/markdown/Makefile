# Build the proposal from the Markdown sources in chapters/.
# All output lands in build/ (gitignored); the source tree stays clean.
#
#   make pdf       -> build/proposal.pdf       (published set: Ch1, Ch3, Validation)
#   make full      -> build/proposal_full.pdf  (every chapter + appendices)
#   make preview CH=chapters/30_model.md  -> build/preview.pdf  (one chapter)
#   make clean     -> remove build/

PANDOC   := pandoc
DEFAULTS := defaults.yaml

# Published set = the current default PDF (all \ifshow* toggles were false).
CHAPTERS := \
	chapters/10_commercial.md \
	chapters/30_model.md \
	chapters/40_validation.md

# Full superset, in reading order. 90_appA begins with a raw \appendix switch,
# so the appendix lettering only appears in this target.
CHAPTERS_FULL := \
	chapters/10_commercial.md \
	chapters/20_development.md \
	chapters/30_model.md \
	chapters/40_validation.md \
	chapters/50_user_manual.md \
	chapters/90_appA_regulatory.md \
	chapters/91_appB_review.md \
	chapters/92_appC_interview.md

.PHONY: pdf full preview clean
pdf: | build
	$(PANDOC) -d $(DEFAULTS) -o build/proposal.pdf $(CHAPTERS)

full: | build
	$(PANDOC) -d $(DEFAULTS) -o build/proposal_full.pdf $(CHAPTERS_FULL)

# Single-chapter quick render: make preview CH=chapters/30_model.md
preview: | build
	$(PANDOC) -d $(DEFAULTS) -o build/preview.pdf $(CH)

build:
	mkdir -p build

clean:
	rm -rf build
