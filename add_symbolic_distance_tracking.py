"""
Code to add symbolic distance tracking to neural-knowledge-assembly-TI-exp.ipynb
This should be integrated into the train_networks function

Add this BEFORE the training loop starts
"""

from collections import defaultdict

def get_symbolic_distance(i, j):
    """Calculate symbolic distance between pair (i, j)"""
    return abs(j - i)

def get_all_pairs_by_distance(items_n):
    """Get all valid pairs organized by symbolic distance"""
    pairs_by_distance = defaultdict(list)
    
    for i in range(items_n):
        for j in range(items_n):
            if i != j:  # Exclude diagonal pairs
                distance = get_symbolic_distance(i, j)
                if items_n - distance > 0:  # Condition: n - sd > 0
                    pairs_by_distance[distance].append((i, j))
    
    return pairs_by_distance

# In the train_networks function, modify the results dictionary to include:
"""
def train_networks(gamma):
    # Log
    results = {
        "train": {
            "losses": np.zeros((seeds_n, training_length)),
            "w1s": np.zeros((seeds_n, training_length, h1_size, items_n)),
            "w2s": np.zeros((seeds_n, training_length, readouts, h1_size)),
            "h1s": np.zeros((seeds_n, training_length, items_n, h1_size)),
            "certainties": np.zeros((seeds_n, training_length, items_n, items_n)),
            "evals": np.zeros((seeds_n, items_n, items_n)),
        },
    }
    
    # ADD THIS: Symbolic distance tracking
    pairs_by_distance = get_all_pairs_by_distance(items_n)
    valid_distances = sorted(pairs_by_distance.keys())
    
    # Create storage for all pairs
    all_pairs_scores = {}
    for distance in valid_distances:
        for pair in pairs_by_distance[distance]:
            key = f"pair_{pair[0]}_{pair[1]}_dist_{distance}"
            all_pairs_scores[key] = np.zeros((seeds_n, training_length))
    
    results["train"]["symbolic_pairs"] = all_pairs_scores
    results["train"]["pairs_by_distance"] = pairs_by_distance
"""

"""
# THEN in the training loop, after each step, add:

# Inside the with torch.no_grad() block where you log results:
with torch.no_grad():
    results["train"]["losses"][seed, training_step] = model.loss.item()
    results["train"]["w1s"][seed, training_step] = model.layer_1.weight.detach().numpy().copy()
    results["train"]["w2s"][seed, training_step] = model.layer_2.weight.detach().numpy().copy()
    results["train"]["h1s"][seed, training_step] = model.extract_h1s()
    results["train"]["certainties"][seed, training_step] = model.pairwise_certainty.a.copy()
    
    # ADD THIS: Evaluate all pairs for symbolic distance tracking
    for distance in valid_distances:
        for pair in pairs_by_distance[distance]:
            i, j = pair
            _, output = model(i, j)
            score = output.item() if readouts == 1 else (output[0].item() - output[1].item())
            key = f"pair_{i}_{j}_dist_{distance}"
            all_pairs_scores[key][seed, training_step] = score
    
    if gamma == 0.:
        results["train"]["training_progress"][seed, training_step] = model.evaluate()
"""

"""
# AFTER training completes, create and save the symbolic distance CSV:

def save_symbolic_distance_csv(results, gamma, filename):
    import csv
    import os
    
    # Create CSV directory
    os.makedirs("run_csvs", exist_ok=True)
    csv_path = os.path.join("run_csvs", filename)
    
    all_pairs_scores = results["train"]["symbolic_pairs"]
    pairs_by_distance = results["train"]["pairs_by_distance"]
    valid_distances = sorted(pairs_by_distance.keys())
    
    # Create header
    header = ["run", "step", "loss"]
    for distance in valid_distances:
        for pair in pairs_by_distance[distance]:
            header.append(f"pair_{pair[0]}_{pair[1]}_dist_{distance}")
    
    # Collect all rows
    all_rows = []
    training_length = results["train"]["losses"].shape[1]
    seeds_n = results["train"]["losses"].shape[0]
    
    for seed in range(seeds_n):
        for step in range(training_length):
            loss = results["train"]["losses"][seed, step]
            row = [seed, step, loss]
            
            # Add all pair scores
            for distance in valid_distances:
                for pair in pairs_by_distance[distance]:
                    key = f"pair_{pair[0]}_{pair[1]}_dist_{distance}"
                    row.append(all_pairs_scores[key][seed, step])
            
            all_rows.append(row)
    
    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    print(f"Saved symbolic distance CSV to: {csv_path}")
    return csv_path

# Call this after training:
# csv_path = save_symbolic_distance_csv(results, gamma, 
#                                       f"knowledge_assembly_gamma_{gamma}_symbolic_distances.csv")
"""

print("Code snippets ready to be integrated into train_networks function")
print("\nSteps to integrate:")
print("1. Add get_symbolic_distance() and get_all_pairs_by_distance() functions")
print("2. Modify results dictionary to include symbolic_pairs storage")
print("3. Add pair evaluation in the logging section")
print("4. Add save_symbolic_distance_csv() function")
print("5. Call save function after training completes")

