# Development Methodology and Controls {#sec:ch-development}

## High-level principles {#sec:high-level-principles}

**Core idea.** Build and enforce a robust system in which a human develops the model and the solution methods, specifies implementation details, and designs the test suites and quality checks.

**High-level design principles:**

- No black-box or LLM-generated design

- Complete and full documentation

- Full reproducibility and record-keeping

- Model validation with known-answer or credible benchmark

This system has been implemented in the latest version of the report (June 20).

## Workflow {#sec:workflow}

The development workflow is organized into the following layers. Each layer states its outcomes, what the AI agents do, and what the human does.

+----------------+----------------+----------------+----------------+
| ::: minipage   | ::: minipage   | ::: minipage   | ::: minipage   |
| Layer          | Outcomes       | What AI agents | What human do  |
| :::            | :::            | do             | :::            |
|                |                | :::            |                |
+:===============+:===============+:===============+:===============+
| . Design and   | Full,          | Drafts the     | Develop the    |
| doc            | human-owned    | code and       | model,         |
|                | design         | environment    | methods, and   |
|                | documentation  | strictly from  | specs. Write   |
|                | (`method.md`,  | your           | the            |
|                | `model.md`)    | documentation, | documentation  |
|                | specifying the | not the AI     | and agent      |
|                | algorithm and  | agent's own    | instructions.  |
|                | implementation | assumptions or |                |
|                | details; an    | hallucination  |                |
|                | agent          |                |                |
|                | instruction    |                |                |
|                | file           |                |                |
|                | (`AGENTS.md`)  |                |                |
+----------------+----------------+----------------+----------------+
| . Write code   | Code written   | Writes the     | Ensure strict  |
|                | to your        | code           | enforcement.   |
|                | documentation  | faithfully to  | Understand     |
|                |                | the doc        | code.          |
+----------------+----------------+----------------+----------------+
| . Test suite   | Unit tests,    | Writes and     | Define the     |
|                | integration    | runs the unit  | tests and the  |
|                | tests,         | and            | pass           |
|                | known-answer   | integration    | thresholds;    |
|                | (oracle)       | tests;         | check that     |
|                | tests, and     | implements and | asserted       |
|                | other          | runs the       | values come    |
|                | correctness    | correctness    | from the spec, |
|                | tests.         | tests you      | not just       |
|                |                | specify        | current output |
+----------------+----------------+----------------+----------------+
| . Code review  | The review     | Checks the     | Read the       |
|                | checklist      | change against | change and     |
|                | (`review.md`), | `review.md`    | resolve flags; |
|                | a second       | and flags      | never let one  |
|                | independent    | issues         | agent both     |
|                | agent, and     |                | write and      |
|                | your own       |                | approve        |
|                | reading        |                |                |
+----------------+----------------+----------------+----------------+
| . Automatic    | The quality    | Runs           | Decide which   |
| gate           | gate: a rule   | formatting,    | checks are     |
|                | that runs the  | type, and test | required; own  |
|                | checks on      | checks; blocks | the rule       |
|                | every proposed | the merge if   |                |
|                | change         | any fail       |                |
+----------------+----------------+----------------+----------------+
| . Runtime      | Built-in       | Stops the run  | Read the run   |
| checks and     | runtime        | on impossible  | logs;          |
| logging        | checks; logs   | values;        | investigate    |
|                | of seeds,      | records logs   | failures on    |
|                | versions, and  | and numbers    | real data      |
|                | key            |                |                |
|                | diagnostics    |                |                |
+----------------+----------------+----------------+----------------+
| . Results      | A short        | Summarizes     | Judge whether  |
| review         | results        | estimates and  | results make   |
|                | checklist      | convergence    | economic and   |
|                |                | diagnostics    | statistical    |
|                |                |                | sense; decide  |
|                |                |                | whether to     |
|                |                |                | ship           |
+----------------+----------------+----------------+----------------+

## Use of AI-tools {#sec:use-of-ai-tools}

The AI agent can mostly write the unit and integration tests, because their expected behavior is the code's own contract already specified in the design documentation and the code. However, the human needs to specify the boundary that applies to agent-written unit and integration tests in `AGENT.md` to ensure proper implementation. For example, the agent must not assert whatever the code currently outputs.

## Validation, safety, and monitoring {#sec:validation-safety-and-monitoring}

There are two types of "correctness" that the full test suite should address. The first type is **whether each piece and the pipeline wiring are correct**, which is covered by:

- **Unit test**: checks that one function does what it is meant to do (a moment calculation, a transition-matrix builder, a data normalizer), including edge cases and error handling.

- **Integration test**: checks that the parts connect and the whole pipeline runs end to end with the right shapes and types.

- **Regression test**: freeze the output of a run already validated, under a fixed seed and pinned versions, and check that future runs still match it. This catches accidental drift; it does not prove the answer is correct.

The second type is the **scientific correctness of the result**. This part requires the human authors to develop the correctness tests carefully:

- **Known-answer test (oracle)**: run the code on a special case with a known analytical solution, or benchmark against credible results from validated methods (e.g., VFI), or simulate data from known true parameters and check the estimator recovers them.

- **Economic property test**: check properties that economic or statistical theory says must hold for any inputs. Change one input and check the output moves in the theory-predicted way (comparative statics).

These correctness tests are mostly model- or method-specific, so they are specified by the human per `model.md` and `method.md`.

Take the risky debt model for example. The known-answer test includes checking that the solved policy and value functions (parameterized by NN) match the solution from grid-based value function iteration. Then use the validated model solver to simulate data with true parameter values, and verify whether the estimation pipeline (SMM or Bayesian) can correctly recover them. To improve confidence on coverage and robustness, these tests need to be repeated for different parameter initializations and simulated batches (RNG seeds).

A `review.md` has been created and will be maintained and updated. The `review.md` includes common high-risk issues, diagnostics, and known fixes that cover:

- TF and TFP usage

- Neural network training

- Estimation

- Data management and matrix operations

- Reproducibility

- Hardware and version compatibility

Note that `review.md` focuses on generic and high-priority issues. Any model- and method-specific issues are covered in `model.md` and `method.md`. The full `review.md` checklist is reproduced in Appendix B.
