import torch
import numpy as np
from nellinetwork import Network, PairwiseCertainty

class SymmetricNetwork(Network):
    """
    Network with symmetric weights between first and second half of items
    
    For 7 items: items 0,1,2 (first half) have symmetric weights to items 3,4,5 (second half)
    Item 6 can be handled separately or as part of second half
    """
    
    def __init__(self, items_n, h1_size, w1_weight_std, w2_weight_std, 
                 non_linearity=torch.relu_, readouts=1, symmetry_mode="strict"):
        """
        Initialize symmetric network
        
        Parameters:
        -----------
        symmetry_mode : str
            "strict" - enforce perfect symmetry during training
            "init_only" - only initialize with symmetry, let training break it
            "soft" - add symmetry regularization loss
        """
        super().__init__(items_n, h1_size, w1_weight_std, w2_weight_std, 
                        non_linearity, readouts)
        
        self.symmetry_mode = symmetry_mode
        self.half_size = items_n // 2  # For 7 items: half_size = 3
        
        # Initialize with symmetric weights
        self._initialize_symmetric_weights(w1_weight_std)
    
    def _initialize_symmetric_weights(self, std):
        """Initialize weights with symmetry between halves"""
        with torch.no_grad():
            # Generate random weights for first half
            first_half_weights = torch.normal(0, std, (self.h1_size, self.half_size))
            
            # Set weights for both halves
            self.layer_1.weight[:, :self.half_size] = first_half_weights
            self.layer_1.weight[:, self.half_size:2*self.half_size] = first_half_weights
            
            # Handle odd number of items (item 6 in 7-item case)
            if self.items_n % 2 == 1:
                # Option 1: Make last item same as last item of second half
                self.layer_1.weight[:, -1] = first_half_weights[:, -1]
                # Option 2: Give last item independent weights
                # self.layer_1.weight[:, -1] = torch.normal(0, std, (self.h1_size,))
    
    def enforce_symmetry(self):
        """Enforce symmetry constraint on weights"""
        if self.symmetry_mode == "strict":
            with torch.no_grad():
                # Average the weights between corresponding positions
                first_half = self.layer_1.weight[:, :self.half_size]
                second_half = self.layer_1.weight[:, self.half_size:2*self.half_size]
                
                # Make them symmetric by averaging
                avg_weights = (first_half + second_half) / 2
                self.layer_1.weight[:, :self.half_size] = avg_weights
                self.layer_1.weight[:, self.half_size:2*self.half_size] = avg_weights
                
                # Handle odd item
                if self.items_n % 2 == 1:
                    self.layer_1.weight[:, -1] = avg_weights[:, -1]
    
    def symmetry_loss(self):
        """Compute symmetry regularization loss"""
        first_half = self.layer_1.weight[:, :self.half_size]
        second_half = self.layer_1.weight[:, self.half_size:2*self.half_size]
        return torch.mean((first_half - second_half)**2)
    
    def forward(self, item_1, item_2):
        """Forward pass with optional symmetry enforcement"""
        if self.symmetry_mode == "strict":
            self.enforce_symmetry()
        
        return super().forward(item_1, item_2)

class SymmetricNetworkV2(Network):
    """
    Alternative implementation: Reduce parameters by sharing weights
    """
    
    def __init__(self, items_n, h1_size, w1_weight_std, w2_weight_std, 
                 non_linearity=torch.relu_, readouts=1):
        # Initialize parent but we'll override layer_1
        super().__init__(items_n, h1_size, w1_weight_std, w2_weight_std, 
                        non_linearity, readouts)
        
        self.half_size = items_n // 2
        
        # Create shared weight matrix for both halves
        self.shared_weights = torch.nn.Parameter(
            torch.normal(0, w1_weight_std, (h1_size, self.half_size))
        )
        
        # Handle odd number of items
        if items_n % 2 == 1:
            self.extra_weight = torch.nn.Parameter(
                torch.normal(0, w1_weight_std, (h1_size, 1))
            )
    
    def get_full_weights(self):
        """Construct full weight matrix from shared weights"""
        if self.items_n % 2 == 0:
            return torch.cat([self.shared_weights, self.shared_weights], dim=1)
        else:
            return torch.cat([self.shared_weights, self.shared_weights, self.extra_weight], dim=1)
    
    def forward(self, item_1, item_2):
        """Forward pass using shared weights"""
        self.item_1 = item_1
        self.item_2 = item_2

        x1 = self._one_hot(item_1)
        x2 = self._one_hot(item_2)
        
        # Use shared weights
        full_weights = self.get_full_weights()
        h1_input = torch.matmul(full_weights, x1) - torch.matmul(full_weights, x2)
        h1 = self.non_linearity(h1_input)

        out = self.layer_2(h1)
        return h1, out

# Training function with symmetric network
def train_symmetric_networks(gamma, symmetry_mode="strict", lambda_sym=0.01):
    """
    Modified training function for symmetric networks
    
    Parameters:
    -----------
    lambda_sym : float
        Regularization strength for soft symmetry (only used if symmetry_mode="soft")
    """
    # ... (copy the existing training setup) ...
    
    # Replace model creation with:
    model = SymmetricNetwork(items_n, h1_size, w1_weight_std, w2_weight_std, 
                           readouts=readouts, symmetry_mode=symmetry_mode)
    
    # In training loop, add symmetry loss if using soft mode:
    if symmetry_mode == "soft":
        total_loss = model.loss + lambda_sym * model.symmetry_loss()
        total_loss.backward()
    else:
        model.loss.backward()
    
    # Rest of training remains the same...
    
    return gamma, results

# Utility functions
def compare_symmetric_vs_regular(items_n=7, h1_size=50):
    """Compare symmetric vs regular network initialization"""
    
    # Regular network
    regular_net = Network(items_n, h1_size, 0.1, 0.1)
    
    # Symmetric network
    symmetric_net = SymmetricNetwork(items_n, h1_size, 0.1, 0.1, symmetry_mode="strict")
    
    print("Regular network weights (first layer):")
    print(regular_net.layer_1.weight.detach().numpy())
    print("\nSymmetric network weights (first layer):")
    print(symmetric_net.layer_1.weight.detach().numpy())
    
    # Check symmetry
    first_half = symmetric_net.layer_1.weight[:, :3].detach().numpy()
    second_half = symmetric_net.layer_1.weight[:, 3:6].detach().numpy()
    print(f"\nSymmetry check - Max difference: {np.max(np.abs(first_half - second_half))}")

if __name__ == "__main__":
    print("Symmetric network implementations loaded!")
    print("Available classes:")
    print("- SymmetricNetwork (enforces symmetry)")
    print("- SymmetricNetworkV2 (parameter sharing)")
    print("Available functions:")
    print("- train_symmetric_networks()")
    print("- compare_symmetric_vs_regular()")

