"""Block 1 (value/policy training) and Block 2 (grid refinement) solver code.

The grid and interpolation helpers here are shared with the VFI benchmark in
``validation/vfi.py`` (oracle-only), but the VFI solver itself lives under
``validation`` so production estimation code never imports it.
"""
