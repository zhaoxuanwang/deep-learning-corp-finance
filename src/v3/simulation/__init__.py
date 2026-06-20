"""Block 2 panel simulation and moment construction (DF26 Sec 4.2, 4.3).

Runs a grid-based solution (VFI benchmark or refined network) forward as a firm
panel, then forms the 11 targeted moments. float64, CPU; stateless RNG keyed per
period (Sec 10).
"""
