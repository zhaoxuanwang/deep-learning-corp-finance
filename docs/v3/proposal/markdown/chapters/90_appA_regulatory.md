```{=latex}
\appendix
```

# Commercial Plan: Regulatory Compliance Map {#sec:appA-regulatory}

**Scope.** This appendix summarizes the laws, rules, and supervisory guidance that a mature version of the product must address when deployed inside a global bank like JPMorgan, across two jurisdictions: the United States, where JPMorgan is supervised as a US bank, and the Hong Kong SAR, where it operates through locally regulated entities. Because the properly-implemented product is a standard statistical and quantitative model rather than a generative AI tool, the governing regime in each jurisdiction is the **quantitative-model (model-risk) regime**, with the AI-specific rules a short overlay.

**What "our product" is, for classification purposes.** The end product is built mostly on the **DF26 framework**, a deep-learning approach to structural estimation, with classical methods (VFI / PFI for solving the model, SMM and Bayesian inference for estimating parameters) as benchmarks. Every method in the stack turns input data into quantitative estimates, and **none is a generative model or an agentic system.** That single fact places the product in the ordinary quantitative-model lane, not the generative-AI lane, in both jurisdictions.

**Use of coding assistants.** The code is written with help from a general coding assistant, but the algorithm and product design are human-owned and the assistant only implements human-specified, human-reviewed plans. This does not change the model's classification; it adds two ordinary controls: all AI-assisted code must be reviewed and tested against the human design, and what may be sent to an external assistant is governed by the bank's AI-tool-use and data-protection policies.

**As-of date.** Verified against sources available as of 17 June 2026.

## The shared compliance baseline and key takeaways {#sec:the-shared-compliance-baseline-and-key-takeaways}

Both the US and Hong Kong regimes demand the same core baseline of a quantitative model; build the model and its paperwork to satisfy these once, and it meets both sides. The same baseline is set out, with jurisdiction sources, in Chapter 1 (Model risk and compliance).

- **A human stays responsible.** The model is decision support; a banker or officer reviews and signs off, and it never decides on its own.

- **No black box.** The model must be explainable to those who rely on it and to independent reviewers, clearly enough to question and challenge.

- **Independent validation before it goes live.** The model must be tested and approved before use, and the builder cannot also be the validator.

- **Write everything down.** Design, assumptions, model choices, data sources, and test results must be documented well enough for a third party to reproduce the work.

- **Test against the real world, not just the fit.** The model must be checked against actual outcomes on data it was not built on, not merely shown to fit its training data.

- **Keep watching it after launch.** Performance must be monitored over time, and the model re-checked or re-estimated when conditions change.

- **Be honest about limitations.** Known weaknesses must be disclosed and use restricted where the model is weak, reporting ranges rather than false precision.

- **Mind the input data.** The data feeding the model must be appropriate, of good quality, and documented.

- **If sold, give the buyer enough to validate it.** Ship a transparency package that lets the buyer's reviewers check the model without handing over proprietary code.

**Match the rigor to the stakes.** Everything above scales with materiality: a low-stakes advisory input a banker double-checks needs lighter treatment than a model wired into capital or pricing. This is why the product is designed to be deployed as a reviewed advisory input rather than an automated engine. A single validation-and-governance package, built to the SR 26-2 standard, satisfies the whole baseline for both jurisdictions.

**Key takeaways:**

1.  **Packaging, not substance.** The US states the whole regime in one document (SR 26-2); Hong Kong assembles the same obligations from several instruments. The cost is mapping and documentation, not a different set of controls.

2.  **The advisory conduct layer is separate and applies in both places.** US SEC / FINRA rules and HK SFC rules govern how the model's output is used in client advice; the common thread is a human adviser who remains responsible.

3.  **The generative-AI rules are an overlay we have engineered around.** Because the product is non-generative, it stays in the ordinary quantitative-model lane; only adding a generative or agentic feature would change that.

## Consolidated list {#sec:consolidated-list}

| Jurisdiction and issuer | Instrument | Type | Binds our product? |
|:---|:---|:---|:---|
| US --- Fed / OCC / FDIC | SR 26-2 / OCC Bulletin 2026-13, Revised MRM Guidance | Quantitative-model (model risk) | **Yes** |
| US --- SEC / FINRA | SEC / FINRA supervision, recordkeeping, fair dealing | Advisory conduct overlay | Conditional (client-facing advice) |
| US --- Fed / OCC / FDIC | Interagency Third-Party Risk Mgmt Guidance (OCC 2023-17) | Vendor onboarding | **Yes, if we are the vendor** |
| US --- Fed / OCC / FDIC; White House | Forthcoming AI RFI; EO 14179 federal posture | AI-specific overlay | No (watch only) |
| HK --- HKMA; Legislature; PRA (benchmark) | Banking Ordinance Cap. 155 (7th Sch.); SPM IC-1, CG-1, CA-G-4, CA-G-3; HKMA 2019 AI principles; PRA SS1/23 (benchmark) | Quantitative-model (model risk) | **Yes (bank entity)** |
| HK --- SFC | Code of Conduct (GP1, 2, 3, 5, 7, 9); CFA Code (Type 6); Internal Control Guidelines | Advisory conduct + model governance | **Yes (advisory entity)** |
| HK --- Legislature / PCPD | PDPO (Cap. 486) + PCPD AI Model Framework | Data privacy | **Yes, if personal data** (CEO product) |
| HK --- SFC; HKMA | SFC GenAI circular (24EC55); HKMA GenAI consumer circular | AI-specific overlay | No (generative / customer-facing only) |

## Watch list {#sec:watch-list}

1.  **US interagency AI RFI.** Not yet issued; will define the regime for generative and agentic AI. Relevant only if we add such a component.

2.  **HKMA updates to the 2019 AI principles.** Expected to be updated over time; track for any move toward firmer model requirements.

3.  **A dedicated HK model-risk standard.** Given international movement (PRA SS1/23) and active HKMA supervision, a more explicit local expectation could emerge; track HKMA SPM updates.

4.  **HK GenAI Sandbox / Sandbox++.** A supervised testing route that could be a channel to pilot the product in Hong Kong; an option, not an obligation.