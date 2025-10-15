import torch
import numpy as np
from nellinetwork import Network

def load_model_from_results(results, gamma, seed=0, step=-1, items_n=7, h1_size=50, 
                           w1_weight_std=0.1, w2_weight_std=0.1, readouts=1):
    """
    Load a model from training results
    
    Parameters:
    -----------
    results : dict
        Results dictionary from training
    gamma : float
        Gamma value to load
    seed : int
        Which seed to load (default: 0)
    step : int
        Which training step to load (-1 for final step)
    items_n, h1_size, w1_weight_std, w2_weight_std, readouts : 
        Network parameters (should match training)
    
    Returns:
    --------
    model : Network
        Loaded network model
    """
    if gamma not in results:
        raise ValueError(f"Gamma {gamma} not found in results")
    
    # Create new model with same architecture
    model = Network(items_n, h1_size, w1_weight_std, w2_weight_std, readouts=readouts)
    
    # Get weights from results
    w1s = results[gamma]["train"]["w1s"][seed, step]  # shape: (h1_size, items_n)
    w2s = results[gamma]["train"]["w2s"][seed, step]  # shape: (readouts, h1_size)
    
    # Load weights into model
    model.layer_1.weight.data = torch.tensor(w1s, dtype=torch.float32)
    model.layer_2.weight.data = torch.tensor(w2s, dtype=torch.float32)
    
    # Load certainty matrix if available
    if "certainties" in results[gamma]["train"]:
        certainties = results[gamma]["train"]["certainties"][seed, step]
        model.pairwise_certainty.a = certainties.copy()
    
    return model

def create_models_from_results(results, gamma_values, seed=0, step=-1, 
                              items_n=7, h1_size=50, w1_weight_std=0.1, 
                              w2_weight_std=0.1, readouts=1):
    """
    Create multiple models from results for comparison
    
    Parameters:
    -----------
    results : dict
        Results dictionary from training
    gamma_values : list
        List of gamma values to load
    seed : int
        Which seed to load (default: 0)
    step : int
        Which training step to load (-1 for final step)
    items_n, h1_size, w1_weight_std, w2_weight_std, readouts : 
        Network parameters (should match training)
    
    Returns:
    --------
    models : list
        List of loaded network models
    model_names : list
        List of model names for plotting
    """
    models = []
    model_names = []
    
    for gamma in gamma_values:
        if gamma in results:
            try:
                model = load_model_from_results(
                    results, gamma, seed, step, items_n, h1_size, 
                    w1_weight_std, w2_weight_std, readouts
                )
                models.append(model)
                model_names.append(f"γ = {gamma}")
            except Exception as e:
                print(f"Failed to load model for gamma={gamma}: {e}")
        else:
            print(f"Gamma {gamma} not found in results")
    
    return models, model_names

def analyze_network_performance(model, training_pairs=None):
    """
    Analyze network performance on specific pairs
    
    Parameters:
    -----------
    model : Network
        Trained network model
    training_pairs : list of tuples
        Pairs to analyze (if None, analyzes all pairs)
    
    Returns:
    --------
    analysis : dict
        Dictionary with performance metrics
    """
    network_output = model.evaluate()
    items_n = model.items_n
    
    analysis = {
        "network_output": network_output,
        "choice_probabilities": torch.sigmoid(torch.tensor(network_output)).numpy(),
        "certainty_matrix": model.pairwise_certainty.a.copy()
    }
    
    if training_pairs is not None:
        pair_outputs = {}
        for i, j in training_pairs:
            pair_outputs[f"({i},{j})"] = network_output[i, j]
        analysis["pair_outputs"] = pair_outputs
    
    # Check transitivity violations
    violations = []
    for i in range(items_n):
        for j in range(items_n):
            for k in range(items_n):
                if i != j and j != k and i != k:
                    # Check if i > j and j > k implies i > k
                    if (network_output[i,j] > 0 and network_output[j,k] > 0 
                        and network_output[i,k] <= 0):
                        violations.append((i, j, k))
    
    analysis["transitivity_violations"] = violations
    analysis["n_violations"] = len(violations)
    
    return analysis

def print_network_summary(model, title="Network Summary"):
    """
    Print a summary of network behavior
    
    Parameters:
    -----------
    model : Network
        Trained network model
    title : str
        Title for the summary
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    network_output = model.evaluate()
    items_n = model.items_n
    
    print(f"Network size: {items_n} items")
    print(f"Hidden layer size: {model.h1_size}")
    print(f"Readouts: {model.readouts}")
    
    print(f"\nNetwork output range: [{np.min(network_output):.3f}, {np.max(network_output):.3f}]")
    
    # Check diagonal (should be near zero)
    diagonal_values = np.diag(network_output)
    print(f"Diagonal values (should be ~0): mean={np.mean(diagonal_values):.3f}, std={np.std(diagonal_values):.3f}")
    
    # Check antisymmetry (output[i,j] ≈ -output[j,i])
    antisymmetry_error = np.mean(np.abs(network_output + network_output.T))
    print(f"Antisymmetry error: {antisymmetry_error:.3f}")
    
    # Transitivity analysis
    analysis = analyze_network_performance(model)
    print(f"Transitivity violations: {analysis['n_violations']}")
    
    # Certainty statistics
    certainty = model.pairwise_certainty.a
    print(f"Certainty range: [{np.min(certainty):.3f}, {np.max(certainty):.3f}]")
    print(f"Mean certainty: {np.mean(certainty):.3f}")

if __name__ == "__main__":
    print("Model utility functions loaded!")
    print("Available functions:")
    print("- load_model_from_results(results, gamma, seed=0, step=-1, ...)")
    print("- create_models_from_results(results, gamma_values, ...)")
    print("- analyze_network_performance(model, training_pairs=None)")
    print("- print_network_summary(model, title='Network Summary')")


