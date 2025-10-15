import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.special import expit
import copy

# Import existing plotting functions
from plotting import plotting_init, matrix_plot, blue, blue2, red, red2

def plot_network_output(model, title="Network Output Matrix", figsize=(8, 6)):
    """
    Plot the network output matrix showing pairwise comparisons
    
    Parameters:
    -----------
    model : Network
        Trained network model
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Get network evaluation
    network_output = model.evaluate()
    items_n = model.items_n
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Plot matrix
    im = matrix_plot(network_output, ax, items_n)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Network Output", rotation=270, labelpad=15)
    cbar.outline.set_edgecolor([0.5] * 3)
    
    # Set title and labels
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Second Item (j)", fontsize=12)
    ax.set_ylabel("First Item (i)", fontsize=12)
    
    # Add interpretation text
    textstr = 'Positive: i > j\nNegative: i < j'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax

def plot_choice_matrix(model, title="Network Choice Probabilities", figsize=(8, 6)):
    """
    Plot the choice probability matrix (sigmoid of network output)
    
    Parameters:
    -----------
    model : Network
        Trained network model
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Get network evaluation and convert to choice probabilities
    network_output = model.evaluate()
    choice_probs = expit(network_output)  # sigmoid function
    
    # Set diagonal to NaN for better visualization
    np.fill_diagonal(choice_probs, np.nan)
    
    items_n = model.items_n
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Plot matrix
    im = matrix_plot(choice_probs, ax, items_n, vmin=0, vmax=1)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("P(choose i over j)", rotation=270, labelpad=15)
    cbar.set_ticks([0, 0.5, 1])
    cbar.outline.set_edgecolor([0.5] * 3)
    
    # Set title and labels
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Second Item (j)", fontsize=12)
    ax.set_ylabel("First Item (i)", fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def plot_network_comparison(models, model_names, figsize=(15, 5)):
    """
    Compare network outputs across multiple models
    
    Parameters:
    -----------
    models : list
        List of trained network models
    model_names : list
        Names for each model
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    if n_models == 1:
        axes = [axes]
    
    # Find global min/max for consistent scaling
    all_outputs = [model.evaluate() for model in models]
    vmin = min(np.min(output) for output in all_outputs)
    vmax = max(np.max(output) for output in all_outputs)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        network_output = model.evaluate()
        items_n = model.items_n
        
        # Plot matrix
        im = matrix_plot(network_output, axes[i], items_n, vmin=vmin, vmax=vmax)
        axes[i].set_title(name, pad=15, fontsize=12, fontweight='bold')
        
        if i == 0:
            axes[i].set_ylabel("First Item (i)", fontsize=10)
        else:
            axes[i].set_ylabel("")
        
        axes[i].set_xlabel("Second Item (j)", fontsize=10)
    
    # Add shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20)
    cbar.set_label("Network Output", rotation=270, labelpad=15)
    cbar.outline.set_edgecolor([0.5] * 3)
    
    plt.tight_layout()
    return fig, axes

def plot_training_progress(results, gamma_value, seed=0, figsize=(12, 8)):
    """
    Plot training progress showing how the network output evolves
    
    Parameters:
    -----------
    results : dict
        Training results dictionary
    gamma_value : float
        Gamma value to plot
    seed : int
        Which seed to plot
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    if gamma_value not in results:
        print(f"Gamma value {gamma_value} not found in results")
        return None, None
    
    # Get training progress data
    if "training_progress" in results[gamma_value]["train"]:
        progress_data = results[gamma_value]["train"]["training_progress"][seed]
        n_steps, items_n, _ = progress_data.shape
        
        # Select time points to show
        time_points = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
        time_labels = ["Start", "25%", "50%", "75%", "End"]
        
        fig, axes = plt.subplots(1, len(time_points), figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # Find global min/max for consistent scaling
        vmin = np.min(progress_data)
        vmax = np.max(progress_data)
        
        for i, (t, label) in enumerate(zip(time_points, time_labels)):
            matrix_data = progress_data[t]
            
            # Plot matrix
            im = matrix_plot(matrix_data, axes[i], items_n, vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{label}\n(Step {t+1})", pad=15, fontsize=10)
            
            if i == 0:
                axes[i].set_ylabel("First Item (i)", fontsize=10)
            else:
                axes[i].set_ylabel("")
            
            axes[i].set_xlabel("Second Item (j)", fontsize=10)
        
        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20)
        cbar.set_label("Network Output", rotation=270, labelpad=15)
        cbar.outline.set_edgecolor([0.5] * 3)
        
        fig.suptitle(f"Training Progress (γ={gamma_value}, seed={seed})", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig, axes
    else:
        print("Training progress data not available (only saved for γ=0)")
        return None, None

def plot_certainty_matrix(model, title="Pairwise Certainty Matrix", figsize=(8, 6)):
    """
    Plot the pairwise certainty matrix
    
    Parameters:
    -----------
    model : Network
        Trained network model
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Get certainty matrix
    certainty_matrix = model.pairwise_certainty.a
    items_n = model.items_n
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Plot matrix
    im = matrix_plot(certainty_matrix, ax, items_n, vmin=0, vmax=1)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Certainty", rotation=270, labelpad=15)
    cbar.set_ticks([0, 0.5, 1])
    cbar.outline.set_edgecolor([0.5] * 3)
    
    # Set title and labels
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Item j", fontsize=12)
    ax.set_ylabel("Item i", fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def plot_comprehensive_analysis(model, title_prefix="Network Analysis", figsize=(16, 12)):
    """
    Create a comprehensive plot showing multiple aspects of network behavior
    
    Parameters:
    -----------
    model : Network
        Trained network model
    title_prefix : str
        Prefix for plot titles
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    items_n = model.items_n
    
    # 1. Raw network output
    network_output = model.evaluate()
    im1 = matrix_plot(network_output, axes[0,0], items_n)
    axes[0,0].set_title("Raw Network Output", fontsize=12, fontweight='bold')
    
    # Add colorbar for raw output
    divider = make_axes_locatable(axes[0,0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("Output", rotation=270, labelpad=10, fontsize=10)
    
    # 2. Choice probabilities
    choice_probs = expit(network_output)
    np.fill_diagonal(choice_probs, np.nan)
    im2 = matrix_plot(choice_probs, axes[0,1], items_n, vmin=0, vmax=1)
    axes[0,1].set_title("Choice Probabilities", fontsize=12, fontweight='bold')
    
    # Add colorbar for choice probabilities
    divider = make_axes_locatable(axes[0,1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label("P(i>j)", rotation=270, labelpad=10, fontsize=10)
    cbar2.set_ticks([0, 0.5, 1])
    
    # 3. Certainty matrix
    certainty_matrix = model.pairwise_certainty.a
    im3 = matrix_plot(certainty_matrix, axes[1,0], items_n, vmin=0, vmax=1)
    axes[1,0].set_title("Pairwise Certainty", fontsize=12, fontweight='bold')
    
    # Add colorbar for certainty
    divider = make_axes_locatable(axes[1,0])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax3)
    cbar3.set_label("Certainty", rotation=270, labelpad=10, fontsize=10)
    cbar3.set_ticks([0, 0.5, 1])
    
    # 4. Hidden layer representations (Euclidean distances)
    h1s = model.extract_h1s()
    from sklearn.metrics import euclidean_distances
    distances = euclidean_distances(h1s)
    im4 = matrix_plot(distances, axes[1,1], items_n)
    axes[1,1].set_title("Hidden Layer Distances", fontsize=12, fontweight='bold')
    
    # Add colorbar for distances
    divider = make_axes_locatable(axes[1,1])
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax4)
    cbar4.set_label("Distance", rotation=270, labelpad=10, fontsize=10)
    
    # Set labels for all subplots
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel("Item j", fontsize=10)
            axes[i,j].set_ylabel("Item i", fontsize=10)
    
    fig.suptitle(f"{title_prefix} - Comprehensive Analysis", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, axes

# Example usage functions
def example_single_network_plot(model):
    """Example of how to plot a single network's output"""
    plotting_init()  # Initialize plotting style
    
    # Plot raw output
    fig1, ax1 = plot_network_output(model, "My Network Output")
    plt.show()
    
    # Plot choice probabilities
    fig2, ax2 = plot_choice_matrix(model, "My Network Choices")
    plt.show()
    
    # Plot comprehensive analysis
    fig3, axes3 = plot_comprehensive_analysis(model, "My Network")
    plt.show()

def example_results_analysis(results, gamma_values=[0.0, 0.1, 1.0]):
    """Example of how to analyze training results"""
    plotting_init()  # Initialize plotting style
    
    # Plot training progress for gamma=0
    if 0.0 in results:
        fig1, axes1 = plot_training_progress(results, 0.0, seed=0)
        if fig1 is not None:
            plt.show()
    
    # Compare different gamma values (you'd need to create models from results)
    # This would require loading the final weights for each gamma
    print("To compare gamma values, load final model weights from results and create Network objects")

if __name__ == "__main__":
    print("Network plotting functions loaded!")
    print("Available functions:")
    print("- plot_network_output(model)")
    print("- plot_choice_matrix(model)")  
    print("- plot_network_comparison(models, model_names)")
    print("- plot_training_progress(results, gamma_value)")
    print("- plot_certainty_matrix(model)")
    print("- plot_comprehensive_analysis(model)")
    print("- example_single_network_plot(model)")
    print("- example_results_analysis(results)")


