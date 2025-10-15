import pandas as pd
import numpy as np

class MarginExtractor:
    """
    Extracts margin information from results dict as produced by experiments.

    Usage:
        extractor = MarginExtractor(results)
        extractor.write_csv("filename.csv")
    """
    def __init__(self, results):
        # If results is a tuple, get the dict
        if isinstance(results, tuple):
            results = results[1]
        self.results = results
        self.training_progress = results["train"]["training_progress"]  # shape: (seeds_n, time_steps, items_n, items_n)
        self.seeds_n, self.time_steps, self.items_n, _ = self.training_progress.shape

    def get_margins(self):
        """
        Returns the 4D array of margins with shape (seeds_n, time_steps, items_n, items_n).
        """
        return self.training_progress

    def write_csv(self, csv_filename):
        """
        Efficiently writes margin info for all (i, j, t) pairs to CSV including diagonals.
        Each row: time_step, i, j, [margins for each seed], mean_margin, std_margin
        """
        import csv

        tp = self.training_progress  # alias for brevity: (seeds_n, time_steps, items_n, items_n)
        seeds_n, time_steps, items_n, _ = tp.shape
        header = ["time_step", "i", "j"] + [f"seed_{seed}" for seed in range(seeds_n)] + ["mean_margin", "std_margin"]

        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Use efficient numpy slicing
            for i in range(items_n):
                for j in range(items_n):
                    # Do NOT skip diagonal (i == j)
                    # tp[:, :, i, j] -> shape (seeds_n, time_steps)
                    margins_ij = tp[:, :, i, j]  # (seeds_n, time_steps)
                    means = margins_ij.mean(axis=0)
                    stds = margins_ij.std(axis=0)
                    for t, (vals, mean, std) in enumerate(zip(margins_ij.T, means, stds)):
                        row = [t, i, j] + list(vals) + [mean, std]
                        writer.writerow(row)

    def load_margin_grid_ti_from_csv(csv_filename, time_step):
        """
        Loads the margins CSV (written by MarginExtractor or similar code) and reconstructs a grid of margin values for a specific time_step.
        Returns:
            margin_grid: (items_n, items_n, seeds_n) array of margins
            var_grid: (items_n, items_n) array of variance across seeds
        """
        df = pd.read_csv(csvpl_filename)
        # Extract seed columns
        seed_cols = [col for col in df.columns if col.startswith("seed_")]
        seeds_n = len(seed_cols)
        items_n = df["i"].max() + 1

        # Get only the rows for the specified time step
        df_t = df[df["time_step"] == time_step]
        margin_grid = np.empty((items_n, items_n, seeds_n))
        var_grid = np.empty((items_n, items_n))
        for _, row in df_t.iterrows():
            i, j = int(row["i"]), int(row["j"])
            vals = np.array([row[col] for col in seed_cols])
            margin_grid[i, j, :] = vals
            var_grid[i, j] = vals.var()
        return margin_grid, var_grid  # margin_grid: (items_n, items_n, seeds_n), var_grid: (items_n, items_n)

    # Example usage:
    # margin_grid, var_grid = load_margin_grid_from_csv(csv_filename, time_step=0)  # (items_n, items_n, seeds_n), (items_n, items_n)


    def load_margin_grid_ti_exp_from_csv(csv_filename, time_step):
        """
        Loads the margins CSV (from MarginExtractor or similar code) and reconstructs a TI-exp grid of margin values
        for a specific time_step, where margin(i, j) = -margin(j, i). Fills both (i, j) and (j, i) entries, flipping sign.
        Returns:
            margin_grid: (items_n, items_n, seeds_n) array
            var_grid: (items_n, items_n) array of variance across seeds, also TI-exp by construction
        """
        df = pd.read_csv(csv_filename)
        # Extract seed columns
        seed_cols = [col for col in df.columns if col.startswith("seed_")]
        seeds_n = len(seed_cols)
        items_n = df["i"].max() + 1

        df_t = df[df["time_step"] == time_step]
        margin_grid = np.empty((items_n, items_n, seeds_n))
        margin_grid[:] = np.nan  # initialize with nan for safety
        var_grid = np.empty((items_n, items_n))
        var_grid[:] = np.nan  # initialize with nan for safety

        for _, row in df_t.iterrows():
            i, j = int(row["i"]), int(row["j"])
            vals = np.array([row[col] for col in seed_cols])
            margin_grid[i, j, :] = vals
            margin_grid[j, i, :] = -vals
            var_val = vals.var()
            var_grid[i, j] = var_val
            var_grid[j, i] = var_val  # variance is symmetric, since var(X) = var(-X)

        return margin_grid, var_grid  # shape: (items_n, items_n, seeds_n), (items_n, items_n)

    # Example usage:
    # ti_margin_grid, var_grid = load_margin_grid_ti_exp_from_csv(csv_filename, time_step=0)  # (items_n, items_n, seeds_n), (items_n, items_n)

    
    
