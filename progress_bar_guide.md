# Progress Bar Guide for Modified Training Function

## What Was Added

The modified `train_networks` function now includes **two levels of progress bars**:

1. **Outer progress bar**: Shows progress across seeds (runs)
2. **Inner progress bar**: Shows progress across training blocks within each seed

## Progress Bar Levels

### Level 1: Seeds Progress Bar
```python
for seed in tqdm(range(seeds_n), desc=f"Training (γ={gamma})", unit="seed"):
```
- Shows: `Training (γ=0.0): 20%|██ | 2/10 [00:30<02:00, 15.0s/seed]`
- Displays overall progress through all random seeds

### Level 2: Blocks Progress Bar (nested)
```python
for block in tqdm(range(training_blocks), desc=f"Seed {seed}", leave=False, unit="block"):
```
- Shows: `Seed 0: 50%|█████ | 4/8 [00:15<00:15, 3.75s/block]`
- Displays progress within current seed
- `leave=False` means it disappears after completing each seed

## Usage Examples

### Single Gamma Training
```python
from tqdm.auto import tqdm

# Run training with progress bars
result = train_networks(0.0)
gamma, result_dict = result

# Save CSV
csv_path = save_symbolic_distance_csv(
    result_dict, 
    gamma, 
    f"knowledge_assembly_gamma_{gamma}_symbolic_distances.csv"
)
```

**Expected Output:**
```
Training (γ=0.0):  10%|█         | 1/10 [00:30<04:30, 30.0s/seed]
  Seed 0: 25%|██▌       | 2/8 [00:07<00:22,  3.7s/block]
```

### Multiple Gammas with Multiprocessing
```python
from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np

gammas = np.concatenate([[0], np.geomspace(1e-4, 1, 69)])

# Three-level progress:
# 1. Outer: gammas being processed
# 2. Middle: seeds within each gamma
# 3. Inner: blocks within each seed

with Pool(20) as p:
    results = list(tqdm(
        p.imap(train_networks, gammas, chunksize=1), 
        total=len(gammas), 
        desc="Training all gammas",
        unit="gamma"
    ))

results_dict = {gamma: result for gamma, result in results}
```

**Expected Output:**
```
Training all gammas:  14%|█▍        | 10/70 [05:00<30:00, 30.0s/gamma]
  Training (γ=0.0001): 30%|███       | 3/10 [01:30<03:30, 30.0s/seed]
    Seed 2: 50%|█████     | 4/8 [00:15<00:15, 3.8s/block]
```

### Simplified Progress (No Nested Bars)

If nested bars are too much, you can simplify:

```python
def train_networks_simple_progress(gamma):
    # ... setup code ...
    
    # Single progress bar for total steps
    total_steps = seeds_n * training_blocks * trials
    
    with tqdm(total=total_steps, desc=f"Training γ={gamma}", unit="step") as pbar:
        for seed in range(seeds_n):
            # ... seed setup ...
            
            for block in range(training_blocks):
                for trial in range(trials):
                    # ... training code ...
                    pbar.update(1)  # Update after each trial
```

**Expected Output:**
```
Training γ=0.0:  35%|███▌      | 3360/9600 [02:00<03:45, 27.7step/s]
```

## Jupyter Notebook Compatibility

The code uses `tqdm.auto` which automatically detects Jupyter and uses widget-based progress bars:

```python
from tqdm.auto import tqdm  # Jupyter-compatible
```

In Jupyter, you'll see nice widget progress bars:
```
Training (γ=0.0) [▓▓▓▓▓▓▓░░░░░░░░░░░░] 30% | 3/10 seeds | 01:30 elapsed | 03:30 remaining
```

## Disabling Progress Bars

If progress bars cause issues:

```python
# Option 1: Use disable parameter
for seed in tqdm(range(seeds_n), disable=True):
    ...

# Option 2: Use dummy range
for seed in range(seeds_n):  # No tqdm wrapper
    ...
```

## Performance Notes

- Progress bars add minimal overhead (<1%)
- Nested bars work well in Jupyter
- For very fast operations, consider `mininterval=1` to reduce updates:
  ```python
  for seed in tqdm(range(seeds_n), mininterval=1.0):
  ```

## Troubleshooting

### Progress bar not updating?
- In multiprocessing, use `chunksize=1` with `imap`
- In Jupyter, use `tqdm.auto` not `tqdm`

### Too many progress bars?
- Set `leave=False` on inner bars
- Or remove inner bars entirely

### Progress bar conflicts with print statements?
- Use `tqdm.write()` instead of `print()`:
  ```python
  tqdm.write(f"Completed seed {seed}")
  ```

