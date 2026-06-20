"""FiLM-conditioned value and policy networks (DF26 Sec 3.3).

State enters a compact trunk; the 8-dim parameter vector enters only through
per-layer FiLM generators that modulate each trunk pre-activation (Eq 25).
Implemented as ``tf.Module`` per Sec 11 so ``trainable_variables`` collects the
trunk and its generators for one optimizer pass.
"""
