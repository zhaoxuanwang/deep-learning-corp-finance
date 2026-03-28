"""v2 data generation package.

Provides deterministic, reproducible dataset generation for offline training.
All randomness is governed by a master seed pair via SeedSchedule.

Public API:
    DataGeneratorConfig  - configuration dataclass
    DataGenerator        - builds trajectory and flattened datasets
    build_iterator       - wraps a dataset dict into a tf.data mini-batch iterator
    SeedSchedule         - deterministic RNG schedule
    SeedScheduleConfig   - SeedSchedule configuration
    VariableID           - stable variable → integer ID mapping
"""

from src.v2.data.rng import SeedSchedule, SeedScheduleConfig, VariableID
from src.v2.data.generator import DataGenerator, DataGeneratorConfig
from src.v2.data.pipeline import build_iterator, validate_dataset_keys

__all__ = [
    "SeedSchedule",
    "SeedScheduleConfig",
    "VariableID",
    "DataGenerator",
    "DataGeneratorConfig",
    "build_iterator",
    "validate_dataset_keys",
]
