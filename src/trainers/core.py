"""
src/trainers/core.py

Core training loop utility for all models.
Replaces the monolithic ExperimentRunner.
"""

import time
import logging
import inspect
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Iterator, Optional, Union

from src.utils.annealing import AnnealingSchedule
from src.trainers.config import OptimizationConfig, AnnealingConfig

logger = logging.getLogger(__name__)

def execute_training_loop(
    trainer: Any,
    dataset: Iterator[Dict[str, tf.Tensor]],
    opt_config: OptimizationConfig,
    anneal_config: AnnealingConfig,
    method_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Execute the common training loop:
    1. Iterate for n_iter
    2. Updates annealing schedule
    3. Calls trainer.train_step()
    4. Logs metrics
    
    Args:
        trainer: Object with .train_step(**kwargs) method
        dataset: Iterator yielding batch dictionaries
        opt_config: Optimization configuration
        anneal_config: Annealing configuration
        method_name: Name of the method for logging
    
    Returns:
        History dictionary containing lists of metrics.
    """
    logger.info(f"Starting Training Loop: Method={method_name}, Iterations={opt_config.n_iter}")
    start_time = time.time()
    
    # Annealing Schedule
    anneal = AnnealingSchedule(
        init_temp=anneal_config.temperature_init,
        min=anneal_config.temperature_min,
        decay=anneal_config.decay,
        schedule=anneal_config.schedule
    )
    
    history: Dict[str, list] = {}
    
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
                 # Find a loss key to print
                 loss_keys = [k for k in metrics.keys() if "loss" in k]
                 loss_str = f"{loss_keys[0]}={metrics[loss_keys[0]]:.4f}" if loss_keys else ""
                 print(f"Iter {i}: {loss_str} T={current_temp:.3f}")
        
        # Update annealing state
        anneal.update()

    elapsed = time.time() - start_time
    logger.info(f"Training finished in {elapsed:.2f}s")
    
    return history
