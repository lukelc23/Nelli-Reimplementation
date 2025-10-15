"""
Complete modified train_networks function with symbolic distance tracking
Copy this to replace your existing train_networks function
"""

from collections import defaultdict
import csv
import os
from tqdm.auto import tqdm  # Use tqdm.auto for Jupyter compatibility

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
                if items_n - distance > 0:
                    pairs_by_distance[distance].append((i, j))
    
    return pairs_by_distance

def train_networks(gamma):
    # Setup pairs by distance
    pairs_by_distance = get_all_pairs_by_distance(items_n)
    valid_distances = sorted(pairs_by_distance.keys())
    
    # Count total pairs
    total_pairs = sum(len(pairs_by_distance[d]) for d in valid_distances)
    print(f"Tracking {total_pairs} pairs across {len(valid_distances)} distances")
    
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
    
    # Symbolic distance tracking storage
    all_pairs_scores = {}
    for distance in valid_distances:
        for pair in pairs_by_distance[distance]:
            key = f"pair_{pair[0]}_{pair[1]}_dist_{distance}"
            all_pairs_scores[key] = np.zeros((seeds_n, training_length))
    
    results["train"]["symbolic_pairs"] = all_pairs_scores
    results["train"]["pairs_by_distance"] = pairs_by_distance
    
    if gamma == 0.:
        results["train"]["training_progress"] = np.zeros((seeds_n, training_length, items_n, items_n))
    
    # Progress bar for seeds
    for seed in tqdm(range(seeds_n), desc=f"Training (γ={gamma})", unit="seed"):
        np.random.seed(seed)
        torch.manual_seed(seed)
    
        # Init Network
        model = Network(items_n, h1_size, w1_weight_std, w2_weight_std, readouts=readouts)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

        training_step = 0
        items_per_context = items_n
        
        # Progress bar for blocks within each seed
        for block in tqdm(range(training_blocks), desc=f"Seed {seed}", leave=False, unit="block"): 
            items_per_context = 7
            p = 5
            q = 3
            training_pairs_norm = np.asarray(list(zip(range(0, items_per_context - 1), range(1, items_per_context))))
            training_pairs_exp = np.asarray([[p,q]])
            training_pairs = np.concatenate([training_pairs_norm, training_pairs_exp], axis=0)
            
            for trial in range(trials):
                # Sample input and target
                random_index = np.random.randint(0, len(training_pairs))
                item_1, item_2 = np.random.choice(training_pairs[random_index], 2, False)
                
                if readouts == 1:
                    if item_1 == p and item_2 == q or item_1 == q and item_2 == p:
                        exception = True
                    else:
                        exception = False
                    if not exception:
                        target = torch.tensor([1. if item_1 > item_2 else -1.])
                    else:
                        target = torch.tensor([-1. if item_1 > item_2 else 1.])
                elif readouts == 2:
                    target = torch.tensor([1., -1.] if item_1 > item_2 else [-1., 1.])

                # Forward propagate and backpropagate
                optimiser.zero_grad()
                _, output = model(item_1, item_2)
                model.loss = criterion(output, target)
                model.loss.backward()
                model.correct(learning_rate, gamma)
                optimiser.step()

                # Log
                with torch.no_grad():
                    results["train"]["losses"][seed, training_step] = model.loss.item()
                    results["train"]["w1s"][seed, training_step] = model.layer_1.weight.detach().numpy().copy()
                    results["train"]["w2s"][seed, training_step] = model.layer_2.weight.detach().numpy().copy()
                    results["train"]["h1s"][seed, training_step] = model.extract_h1s()
                    results["train"]["certainties"][seed, training_step] = model.pairwise_certainty.a.copy()
                    
                    # Evaluate all pairs for symbolic distance tracking
                    for distance in valid_distances:
                        for pair in pairs_by_distance[distance]:
                            i, j = pair
                            _, pair_output = model(i, j)
                            if readouts == 1:
                                score = pair_output.item()
                            else:
                                score = pair_output[0].item() - pair_output[1].item()
                            key = f"pair_{i}_{j}_dist_{distance}"
                            all_pairs_scores[key][seed, training_step] = score
                    
                    if gamma == 0.:
                        results["train"]["training_progress"][seed, training_step] = model.evaluate()
                
                training_step += 1
        
        # Evaluate
        with torch.no_grad():
            results["train"]["evals"][seed] = model.evaluate()
        
    return gamma, results

def save_symbolic_distance_csv(results, gamma, csv_filename):
    """Save symbolic distance tracking to CSV"""
    os.makedirs("run_csvs", exist_ok=True)
    csv_path = os.path.join("run_csvs", csv_filename)
    
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

# ============================================================================
# Usage Examples
# ============================================================================

# Example 1: Single gamma with progress bar
# ------------------------------------------
# results = train_networks(0.0)
# gamma, result_dict = results
# csv_path = save_symbolic_distance_csv(result_dict, gamma, 
#                                       f"knowledge_assembly_gamma_{gamma}_symbolic_distances.csv")

# Example 2: Multiple gammas with multiprocessing and progress bar
# ------------------------------------------------------------------
# from multiprocessing import Pool
# from tqdm.auto import tqdm
# 
# gammas = np.concatenate([[0], np.geomspace(1e-4, 1, 69)])
# 
# with Pool(20) as p:
#     results = list(tqdm(
#         p.imap(train_networks, gammas, chunksize=1), 
#         total=len(gammas), 
#         desc="Training all gammas",
#         unit="gamma"
#     ))
# 
# results_dict = {gamma: result for gamma, result in results}
# 
# # Save symbolic distance CSVs for selected gammas
# for gamma in [0.0, 0.1, 1.0]:
#     if gamma in results_dict:
#         csv_path = save_symbolic_distance_csv(
#             results_dict[gamma], 
#             gamma, 
#             f"knowledge_assembly_gamma_{gamma}_symbolic_distances.csv"
#         )

# Example 3: Progress bar with concurrent.futures (alternative)
# --------------------------------------------------------------
# from concurrent.futures import ProcessPoolExecutor
# from tqdm.auto import tqdm
# 
# gammas = [0.0, 0.1, 0.5, 1.0]
# 
# with ProcessPoolExecutor(max_workers=4) as executor:
#     futures = {executor.submit(train_networks, g): g for g in gammas}
#     
#     results = {}
#     for future in tqdm(futures, desc="Training", unit="gamma"):
#         gamma, result = future.result()
#         results[gamma] = result
#         
#         # Save immediately after each gamma completes
#         csv_path = save_symbolic_distance_csv(
#             result, 
#             gamma, 
#             f"knowledge_assembly_gamma_{gamma}_symbolic_distances.csv"
#         )

