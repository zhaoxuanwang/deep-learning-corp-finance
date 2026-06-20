"""Pure economic-model primitives (DF26 Sec 1).

No networks, no solver. Functions are dtype-agnostic (they follow their tensor
inputs) so the same code serves float32 training and float64 exact computation.
The exact and smoothed (training-time) dividend share one implementation.
"""
