
import pytest
from collections import deque
from src.trainers.stopping import ConvergenceChecker, StoppingState
from src.trainers.config import EarlyStoppingConfig

def test_initialization():
    """Verify initialization and buffer sizing."""
    checker = ConvergenceChecker(
        method='lr',
        n_anneal=10,
        lr_window=5,
        ma_window=3
    )
    assert checker.ma_window == 3
    # Buffer should be at least (lr_window + ma_window) * 2 = (5 + 3) * 2 = 16
    assert checker._lr_loss_buffer.maxlen == 16

def test_ma_calculation_logic():
    """Verify that the moving average logic (internal helper) works correctly."""
    # We can test this by running _check_lr_criteria with controlled inputs
    # Let lr_window=5, ma_window=3. 
    # Buffer needs 8 samples to start checking.
    
    checker = ConvergenceChecker(
        method='lr',
        n_anneal=0,
        lr_window=5,
        ma_window=3,
        lr_epsilon=0.1, # 10%
        patience=1
    )
    
    # Fill buffer with constant values 10.0
    # Then current MA = 10, past MA = 10. Rel diff = 0. Should stop.
    metrics = {'loss_LR': 10.0}
    
    # Add 7 samples. len=7 < 5+3=8. Should return False.
    for _ in range(7):
        assert checker.check(step=100, metrics=metrics) is False
        
    # Add 8th sample. Now len=8. Should check.
    # Diff is 0, so < 0.1. Should return True.
    assert checker.check(step=100, metrics=metrics) is True
    
def test_ma_smoothing_effect():
    """Verify that MA enables stopping on noisy but flat plateau."""
    # lr_window=5, ma_window=5
    checker = ConvergenceChecker(
        method='lr',
        n_anneal=0,
        lr_window=5,
        ma_window=5,
        lr_epsilon=0.01
    )
    
    # Create a noisy sequence fluctuating around 100.
    # [101, 99, 101, 99, ...] -> Average is 100.
    # Past window (indices 0-4) avg 100.
    # Current window (indices 5-9) avg 100.
    # Rel diff should be 0.
    
    vals = [101.0, 99.0] * 10
    
    metrics = {}
    stopped = False
    for v in vals:
        metrics['loss_LR'] = v
        if checker.check(step=100, metrics=metrics):
            stopped = True
            break
            
    assert stopped is True

def test_ma_prevents_premature_stopping():
    """Verify that MA prevents stopping when improvement is real but last step is noisy up."""
    # lr_window=5, ma_window=3.
    # Past: [100, 100, 100] -> Avg 100.
    # Gap...
    # Current: [90, 90, 95] -> Avg 91.66.
    # Improvement is (91.66 - 100)/100 = -0.083. Abs = 0.083.
    # If epsilon is 0.05, 0.083 > 0.05, so it should NOT stop (improvement is significant).
    # But if we just looked at raw lag: 95 vs 100 might be < epsilon if epsilon was large, 
    # but here let's stick to the smoothing check.
    
    checker = ConvergenceChecker(
        method='lr',
        n_anneal=0,
        lr_window=5,
        ma_window=3,
        lr_epsilon=0.05 
    )
    
    # 1. Past data: 100, 100, 100
    for _ in range(3):
        checker._lr_loss_buffer.append(100.0)
        
    # 2. Fill gap (lr_window=5, so gap is 5 - ma_window ?)
    # Actually, let's just carefully look at indices logic.
    # Buffer: [P1, P2, P3, ..., C1, C2, C3]
    # Current MA uses C1..C3.
    # Past MA uses window shifted by lr_window=5.
    # So if indices are 0..N-1.
    # Current is N-1, N-2, N-3.
    # Past is N-1-5, N-2-5, N-3-5 => N-6, N-7, N-8.
    # So we need N >= 8 samples total.
    
    # Let's construct buffer manually for clarity
    # Indices: 0, 1, 2 (Past MA target) ... 5, 6, 7 (Current MA target)
    
    vals = [100.0]*5 + [90.0, 90.0] 
    for v in vals:
        checker._lr_loss_buffer.append(v)
        
    # Now append last value 95.0. 
    # Buffer will be [100, 100, 100, 100, 100, 90, 90, 95]
    # Current MA (last 3): (90+90+95)/3 = 275/3 = 91.67
    # Past MA (shifted by 5): Indices -6, -7, -8 from end.
    # -6 is 100. -7 is 100. -8 is 100. Avg = 100.
    
    metrics = {'loss_LR': 95.0} # This adds the 8th element
    
    # Check
    should_stop = checker.check(step=100, metrics=metrics)
    
    # Check calculation details
    # Rel = (91.67 - 100) / 100 = -0.0833
    # Abs(rel) = 0.0833 > 0.05 (epsilon)
    # So it should NOT stop.
    assert should_stop is False
    
