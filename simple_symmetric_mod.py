# Simple modification to make existing network symmetric
import torch
import numpy as np

def make_network_symmetric(model, items_n=7):
    """
    Make an existing network's first layer weights symmetric
    between first and second half of items
    """
    half_size = items_n // 2  # 3 for 7 items
    
    with torch.no_grad():
        # Get current weights
        weights = model.layer_1.weight
        
        # Average corresponding positions
        first_half = weights[:, :half_size]  # columns 0,1,2
        second_half = weights[:, half_size:2*half_size]  # columns 3,4,5
        
        # Make symmetric by averaging
        avg_weights = (first_half + second_half) / 2
        
        # Set both halves to the average
        model.layer_1.weight[:, :half_size] = avg_weights
        model.layer_1.weight[:, half_size:2*half_size] = avg_weights
        
        # Handle the 7th item (index 6) - make it same as item 2 and 5
        if items_n % 2 == 1:
            model.layer_1.weight[:, -1] = avg_weights[:, -1]

def enforce_symmetry_during_training(model, items_n=7):
    """
    Call this after each gradient step to maintain symmetry
    """
    make_network_symmetric(model, items_n)

# Example usage in training loop:
def modified_training_step(model, optimiser, item_1, item_2, target, gamma, learning_rate):
    """
    Training step with symmetry enforcement
    """
    # Forward pass
    optimiser.zero_grad()
    _, output = model(item_1, item_2)
    model.loss = torch.nn.MSELoss()(output, target)
    model.loss.backward()
    
    # Apply corrections (existing knowledge assembly)
    model.correct(learning_rate, gamma)
    
    # Update weights
    optimiser.step()
    
    # Enforce symmetry after weight update
    enforce_symmetry_during_training(model)

# Alternative: Initialize with symmetry only
def initialize_symmetric_weights(model, items_n=7, std=0.1):
    """
    Initialize network with symmetric weights but don't enforce during training
    """
    half_size = items_n // 2
    h1_size = model.h1_size
    
    with torch.no_grad():
        # Generate weights for first half only
        first_half_weights = torch.normal(0, std, (h1_size, half_size))
        
        # Set symmetric weights
        model.layer_1.weight[:, :half_size] = first_half_weights
        model.layer_1.weight[:, half_size:2*half_size] = first_half_weights
        
        # Handle odd item
        if items_n % 2 == 1:
            model.layer_1.weight[:, -1] = first_half_weights[:, -1]

if __name__ == "__main__":
    print("Simple symmetric modifications loaded!")
    print("Functions:")
    print("- make_network_symmetric(model)")
    print("- enforce_symmetry_during_training(model)")  
    print("- initialize_symmetric_weights(model)")
    print("- modified_training_step(...)")

