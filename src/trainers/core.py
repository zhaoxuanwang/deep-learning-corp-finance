"""
src/trainers/core.py

Core training loop utility for all models.
Replaces the monolithic ExperimentRunner.

References:
    report_brief.md lines 723-784: Convergence and Stopping Criteria
    report_brief.md lines 773-784: API Usage
"""

import time
import logging
import inspect
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Iterator, Optional, Union, Callable

from src.utils.annealing import AnnealingSchedule
from src.trainers.config import OptimizationConfig, AnnealingConfig
from src.trainers.stopping import ConvergenceChecker, create_convergence_checker

logger = logging.getLogger(__name__)


def execute_training_loop(
    trainer: Any,
    dataset: Iterator[Dict[str, tf.Tensor]],
    opt_config: OptimizationConfig,
    anneal_config: AnnealingConfig,
    method_name: str = "unknown",
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    validation_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Execute the common training loop with optional early stopping.

    Training modes (report_brief.md lines 773-777):
    - Debug/Demo: early_stopping=None/disabled. Runs exactly n_iter steps.
    - Full Mode: early_stopping=enabled. Terminates when validation metrics
      satisfy method-specific convergence criteria.

    Loop steps:
    1. Validate dataset format matches method requirements
    2. Create annealing schedule and convergence checker
    3. For each iteration:
       a. Update temperature
       b. Get batch and run train_step()
       c. Log metrics
       d. If eval_freq: evaluate on validation set
       e. Check convergence criteria (post-annealing gatekeeper)
    4. Return history

    Args:
        trainer: Object with .train_step(**kwargs) method
        dataset: Iterator yielding batch dictionaries
        opt_config: Optimization configuration
        anneal_config: Annealing configuration
        method_name: Name of the method for logging
        validation_data: Optional validation dataset for convergence checking
        validation_fn: Optional function to compute validation metrics.
                       Signature: validation_fn(trainer, batch, temperature) -> Dict[str, float]

    Returns:
        History dictionary containing lists of metrics, plus 'stopped_early' flag.

    Raises:
        ValueError: If dataset format doesn't match method requirements
    """
    early_stopping = opt_config.early_stopping
    logger.info(
        f"Starting Training Loop: Method={method_name}, Iterations={opt_config.n_iter}, "
        f"EarlyStopping={'enabled' if early_stopping and early_stopping.enabled else 'disabled'}"
    )

    # ===================================================================
    # Dataset Format Validation
    # ===================================================================
    # Peek at first batch to validate format without consuming it
    first_batch = next(dataset)

    # Detect method family from name (e.g., "basic_lr" -> "lr")
    method_parts = method_name.lower().split('_')
    method_family = None
    for part in ['lr', 'er', 'br']:
        if part in method_parts:
            method_family = part
            break

    if method_family is None:
        logger.warning(f"Could not detect method family from '{method_name}', skipping format validation")
    else:
        # Validate based on method family
        batch_keys = set(first_batch.keys())

        if method_family == 'lr':
            # LR methods require trajectory data
            required_keys = {'k0', 'z_path'}
            if not required_keys.issubset(batch_keys):
                raise ValueError(
                    f"Method '{method_name}' (LR family) requires TRAJECTORY data with keys {required_keys}.\n"
                    f"Found keys: {batch_keys}\n"
                    f"Use DataGenerator.get_training_dataset() or get_training_batches() for LR methods."
                )

        elif method_family in ['er', 'br']:
            # ER/BR methods require flattened data
            required_keys = {'k', 'z', 'z_next_main', 'z_next_fork'}
            if not required_keys.issubset(batch_keys):
                raise ValueError(
                    f"Method '{method_name}' ({method_family.upper()} family) requires FLATTENED data with keys {required_keys}.\n"
                    f"Found keys: {batch_keys}\n"
                    f"Use DataGenerator.get_flattened_training_dataset() for ER/BR methods."
                )

        logger.info(f"Dataset format validation passed for {method_family.upper()} method")

    # Create new iterator that includes the first batch
    # Use a generator to prepend the first batch
    def prepend_first_batch(source_iter):
        yield first_batch
        yield from source_iter

    dataset = prepend_first_batch(dataset)

    start_time = time.time()
    
    # Annealing Schedule
    anneal = AnnealingSchedule(
        init_temp=anneal_config.temperature_init,
        min_temp=anneal_config.temperature_min,
        decay_rate=anneal_config.decay,
        # buffer defaults to 0.25, which is what we want
    )

    # Convergence Checker (early stopping)
    convergence_checker = create_convergence_checker(
        method_name=method_name,
        n_anneal=anneal.n_anneal,
        early_stopping_config=early_stopping
    )

    if convergence_checker:
        logger.info(f"Convergence checker created. N_anneal={anneal.n_anneal}, Patience={early_stopping.patience}")

    history: Dict[str, list] = {}
    stopped_early = False

    for i in range(1, opt_config.n_iter + 1):
        # 1. Update Temperature
        current_temp = anneal.value
        
        # 2. Get Batch
        try:
            batch_states = next(dataset)
        except StopIteration:
            logger.info("Dataset exhausted before n_iter.")
            break
        
        # 3. Prepare Arguments for train_step
        # Map dataset keys (k0, z0, b0) to standardized keys (k, z, b) expected by trainers
        train_args = {}
        if "k0" in batch_states: train_args["k"] = batch_states["k0"]
        if "z0" in batch_states: train_args["z"] = batch_states["z0"] 
        if "b0" in batch_states: train_args["b"] = batch_states["b0"]
        
        # Combine all potential args
        step_args = {
            **batch_states,
            **train_args,
            "temperature": current_temp
        }
        
        # Filter args based on trainer.train_step signature
        # This allows trainers to only request what they need (e.g. just k, z)
        sig = inspect.signature(trainer.train_step)
        valid_keys = sig.parameters.keys()
        
        # Check if function accepts **kwargs
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        
        if has_kwargs:
            filtered_args = step_args
        else:
            filtered_args = {k: v for k, v in step_args.items() if k in valid_keys}

        # 4. Train Step
        metrics = trainer.train_step(**filtered_args)
        
        # 5. Logging
        if i % opt_config.log_every == 0 or i == 1:
            rec = {"iteration": i, "temperature": current_temp, **metrics}
            
            # Remove tensor objects from history (e.g. huge arrays) to keep it light
            if "terminal_states" in rec:
                del rec["terminal_states"]
            
            # Convert numpy/tf scalars to python float
            for k, v in rec.items():
                if hasattr(v, "numpy"):
                    rec[k] = float(v)
                elif isinstance(v, (np.float32, np.float64)):
                    rec[k] = float(v)
            
            # Append to history
            for k, v in rec.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)
            
            # Console Log (less frequent)
            if i % (opt_config.log_every * 10) == 0:
                 # Find all loss keys and format them
                 loss_keys = sorted([k for k in metrics.keys() if "loss" in k.lower()])

                 if loss_keys:
                     # For methods with multiple losses (BR, Risky BR), show all
                     # For single-loss methods (LR, ER), show just one
                     if len(loss_keys) == 1:
                         loss_str = f"{loss_keys[0]}={metrics[loss_keys[0]]:.4f}"
                     else:
                         # Multiple losses: show all with abbreviated names
                         # Remove 'loss_' prefix for cleaner display
                         loss_parts = [f"{k.replace('loss_', '')}={metrics[k]:.4f}" for k in loss_keys]
                         loss_str = " ".join(loss_parts)
                 else:
                     loss_str = ""

                 logger.info(f"Iter {i}: {loss_str} Temp (log10)={np.log10(current_temp):.2f}")
        
        # Update annealing state
        anneal.update()

        # ===================================================================
        # Convergence Checking (Early Stopping)
        # Reference: report_brief.md lines 779-784
        # ===================================================================
        if convergence_checker is not None and early_stopping is not None:
            # Only check at eval_freq intervals
            if i % early_stopping.eval_freq == 0:
                # Get validation metrics
                if validation_data is not None and validation_fn is not None:
                    # Evaluate on validation set
                    val_metrics = validation_fn(trainer, validation_data, current_temp)
                    validation_metrics = val_metrics
                else:
                    # Use training metrics as proxy (not ideal but functional)
                    validation_metrics = metrics

                # Check convergence
                if convergence_checker.check(i, validation_metrics):
                    stopped_early = True
                    logger.info(f"Early stopping at iteration {i}")
                    break

    elapsed = time.time() - start_time
    logger.info(f"Training finished in {elapsed:.2f}s (stopped_early={stopped_early})")

    # Add metadata to history
    history['stopped_early'] = stopped_early
    history['final_iteration'] = i
    history['n_anneal'] = anneal.n_anneal

    return history
