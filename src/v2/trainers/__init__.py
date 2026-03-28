"""Trainer package.

Keep this module import-light so ``import src.v2.trainers`` never pulls in
method implementations as a side effect. Import concrete trainers from their
module paths, for example ``src.v2.trainers.lr`` or ``src.v2.trainers.shac``.
"""

__all__: list[str] = []
