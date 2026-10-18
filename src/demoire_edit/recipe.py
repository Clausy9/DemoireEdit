"""Paper recipe validation and scheduling, independent of sampling backend."""
import json
import math
from pathlib import Path
LOSS_COMPONENTS = frozenset({'velocity', 'latent_mse', 'l1', 'lpips', 'ffl'})

def condition_shift_steps(ratio, steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('ratio must be finite and between 0 and 1')
    return math.ceil(ratio * steps)

def condition_schedule(ratio, steps):
    switch = condition_shift_steps(ratio, steps)
    return ['stage1_output'] * switch + ['original_input'] * (steps - switch)

def validate_recipe(recipe):
    loss = recipe['training']['loss']
    if set(loss) != LOSS_COMPONENTS:
        raise ValueError('requires exactly velocity, latent_mse, l1, lpips and ffl')
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in loss.values()):
        raise ValueError('loss weights must be finite and positive')
    stage2 = recipe['inference']['stage2']
    condition_shift_steps(stage2['condition_ratio'], stage2['steps'])
    if stage2['latent_initialization'] != 'noise':
        raise ValueError('selected route requires Stage 2 noise initialization')
    return recipe

def load_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text()))
