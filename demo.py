#!/usr/bin/env python3
"""
Symmetry Calculation for Point Clusters - Demo Script

This demo demonstrates how to calculate symmetry scores for clusters of points,
which can be useful as a loss term in data clustering algorithms to encourage
symmetric cluster formations.

The script shows:
1. Generating synthetic symmetric and asymmetric clusters
2. Calculating different types of symmetry (reflection, rotational, point)
3. Using symmetry as a loss function in clustering
4. Visualizing symmetry scores for different cluster configurations
5. Practical application in K-means clustering with symmetry regularization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings

# Import from the available modules
from lib.core import (
    calculate_reflection_symmetry,
    calculate_rotational_symmetry,
    calculate_point_symmetry,
    calculate_overall_symmetry,
    symmetry_loss
)
from lib.utils import (
    generate_symmetric_cluster,
    normalize_points,
    compute_cluster_metrics,
    find_symmetry_plane,
    split_by_symmetry_plane,
    calculate_symmetry_score_matrix,
    estimate_symmetry_order
)

warnings.filterwarnings('ignore')


def demo_basic_symmetry_calculation():
    """
    Demo 1: Basic symmetry calculation for different cluster types.
    """
    print("=" * 80)
    print("DEMO 1: Basic Symmetry Calculation")
    print("=" * 80)
    
    # Generate different types of symmetric clusters
    print("\n1. Generating symmetric clusters...")
    
    # Reflection symmetric cluster
    reflection_cluster = generate_symmetric_cluster(
        n_points=100,
        n_dims=2,
        symmetry_type='reflection',
        noise_level=0.05,
        random_state=42
    )
    
    # Rotational symmetric cluster
    rotational_cluster = generate_symmetric_cluster(
        n_points=100,
        n_dims=2,
        symmetry_type='rotational',
        noise_level=0.05,
        random_state=42
    )
    
    # Point symmetric cluster
    point_cluster = generate_symmetric_cluster(
        n_points=100,
        n_dims=2,
        symmetry_type='point',
        noise_level=0.05,
        random_state=42
    )
    
    # Random asymmetric cluster
    random_cluster = np.random.randn(100, 2) * 2
    random_cluster[:, 0] += 3  # Shift to make it asymmetric
    
    # Calculate symmetry scores
    print("\n2. Calculating symmetry scores...")
    
    clusters = {
        'Reflection Symmetric': reflection_cluster,
        'Rotational Symmetric': rotational_cluster,
        'Point Symmetric': point_cluster,
        'Random (Asymmetric)': random_cluster
    }
    
    for name, cluster in clusters.items():
        print(f"\n{name}:")
        
        # Reflection symmetry
        refl_score = calculate_reflection_symmetry(cluster, optimize=True)
        print(f"  Reflection Symmetry: {refl_score:.4f}")
        
        # Rotational symmetry (2-fold)
        rot_score = calculate_rotational_symmetry(cluster, n_fold=2, optimize=True)
        print(f"  Rotational Symmetry (2-fold): {rot_score:.4f}")
        
        # Point symmetry
        point_score = calculate_point_symmetry(cluster)
        print(f"  Point Symmetry: {point_score:.4f}")
        
        # Overall symmetry
        overall_score = calculate_overall_symmetry(cluster)
        print(f"  Overall Symmetry: {overall_score:.4f}")
        
        # Symmetry loss (1 - symmetry score)
        loss = symmetry_loss(cluster, symmetry_type='overall')
        print(f"  Symmetry Loss: {loss:.4f}")
    
    # Visualize clusters
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (name, cluster) in enumerate(clusters.items()):
        ax = axes[idx]
        ax.scatter(cluster[:, 0], cluster[:, 1], alpha=0.6, s=30)
        ax.set_title(f"{name}\nOverall Symmetry: {calculate_overall_symmetry(cluster):.3f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add symmetry plane if reflection symmetry is high
        if calculate_reflection_symmetry(cluster, optimize=True) > 0.7:
            normal, point = find_symmetry_plane(cluster)
            # Draw symmetry line
            if cluster.shape[1] == 2:
                x_range = np.array([cluster[:, 0].min() - 1, cluster[:, 0].max() + 1])
                if abs(normal[0]) > 1e-6:
                    slope = -normal[1] / normal[0]
                    intercept = point[1] - slope * point[0]
                    y_range = slope * x_range + intercept
                    ax.plot(x_range, y_range, 'r--', linewidth=2, label='Symmetry Plane')
                    ax.legend()
    
    plt.tight_layout()
    plt.savefig('symmetry_clusters.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'symmetry_clusters.png'")
    plt.close()


def demo_symmetry_order_estimation():
    """
    Demo 2: Estimate the order of rotational symmetry.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Rotational Symmetry Order Estimation")
    print("=" * 80)
    
    # Generate clusters with different rotational symmetries
    symmetry_orders = [2, 3, 4, 6]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, order in enumerate(symmetry_orders):
        print(f"\n{order}-fold Rotational Symmetry:")
        
        # Generate points with n-fold symmetry
        n_per_arm = 20
        angles = np.linspace(0, 2 * np.pi, order, endpoint=False)
        points_list = []
        
        for angle in angles:
            # Create one arm
            r = np.linspace(0.5, 2, n_per_arm)
            theta = np.random.randn(n_per_arm) * 0.1 + angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points_list.append(np.column_stack([x, y]))
        
        cluster = np.vstack(points_list)
        
        # Estimate symmetry order
        estimated_order = estimate_symmetry_order(cluster, max_order=8)
        print(f"  True order: {order}")
        print(f"  Estimated order: {estimated_order}")
        
        # Calculate symmetry scores for different orders
        scores = []
        test_orders = range(2, 9)
        for test_order in test_orders:
            score = calculate_rotational_symmetry(cluster, n_fold=test_order, optimize=True)
            scores.append(score)
            if test_order <= 6:
                print(f"  {test_order}-fold symmetry score: {score:.4f}")
        
        # Visualize
        ax = axes[idx]
        ax.scatter(cluster[:, 0], cluster[:, 1], alpha=0.6, s=30)
        ax.set_title(f"{order}-fold Symmetry (Estimated: {estimated_order})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('rotational_symmetry_orders.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'rotational_symmetry_orders.png'")
    plt.close()


def demo_symmetry_in_clustering():
    """
    Demo 3: Using symmetry as a regularization term in K-means clustering.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Symmetry-Regularized Clustering")
    print("=" * 80)
    
    # Generate data with multiple clusters
    np.random.seed(42)
    
    # Create 3 clusters with different symmetry properties
    cluster1 = generate_symmetric_cluster(50, 2, 'reflection', noise_level=0.1) + np.array([0, 0])
    cluster2 = generate_symmetric_cluster(50, 2, 'rotational', noise_level=0.1) + np.array([5, 0])
    cluster3 = np.random.randn(50, 2) * 0.8 + np.array([2.5, 4])
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    print("\n1. Standard K-means clustering...")
    
    # Simple K-means implementation
    def kmeans_with_symmetry(data, n_clusters, symmetry_weight=0.0, n_iterations=50):
        """K-means with optional symmetry regularization."""
        # Initialize centroids
        indices = np.random.choice(len(data), n_clusters, replace=False)
        centroids = data[indices].copy()
        
        for iteration in range(n_iterations):
            # Assign points to nearest centroid
            distances = np.zeros((len(data), n_clusters))
            for i in range(n_clusters):
                distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    # Standard centroid update
                    new_centroids[i] = cluster_points.mean(axis=0)
                    
                    # Apply symmetry regularization
                    if symmetry_weight > 0:
                        # Calculate symmetry loss for this cluster
                        sym_loss = symmetry_loss(cluster_points, symmetry_type='overall')
                        
                        # Adjust centroid to improve symmetry (simple gradient step)
                        # This is a simplified approach for demonstration
                        if sym_loss > 0.5:  # If cluster is not very symmetric
                            # Move centroid slightly toward cluster center
                            cluster_center = cluster_points.mean(axis=0)
                            new_centroids[i] = (1 - symmetry_weight * 0.1) * new_centroids[i] + \
                                             (symmetry_weight * 0.1) * cluster_center
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            
            centroids = new_centroids
        
        return labels, centroids
    
    # Run standard K-means
    labels_standard, centroids_standard = kmeans_with_symmetry(data, n_clusters=3, symmetry_weight=0.0)
    
    # Run symmetry-regularized K-means
    labels_symmetric, centroids_symmetric = kmeans_with_symmetry(data, n_clusters=3, symmetry_weight=0.3)
    
    # Calculate symmetry scores for each cluster
    print("\n2. Analyzing cluster symmetry...")
    
    print("\nStandard K-means:")
    for i in range(3):
        cluster_points = data[labels_standard == i]
        sym_score = calculate_overall_symmetry(cluster_points)
        print(f"  Cluster {i+1}: {len(cluster_points)} points, Symmetry: {sym_score:.4f}")
    
    print("\nSymmetry-Regularized K-means:")
    for i in range(3):
        cluster_points = data[labels_symmetric == i]
        sym_score = calculate_overall_symmetry(cluster_points)
        print(f"  Cluster {i+1}: {len(cluster_points)} points, Symmetry: {sym_score:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard K-means
    ax = axes[0]
    for i in range(3):
        cluster_points = data[labels_standard == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  alpha=0.6, s=30, label=f'Cluster {i+1}')
    ax.scatter(centroids_standard[:, 0], centroids_standard[:, 1], 
              c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
    ax.set_title('Standard K-means')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Symmetry-regularized K-means
    ax = axes[1]
    for i in range(3):
        cluster_points = data[labels_symmetric == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  alpha=0.6, s=30, label=f'Cluster {i+1}')
    ax.scatter(centroids_symmetric[:, 0], centroids_symmetric[:, 1], 
              c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
    ax.set_title('Symmetry-Regularized K-means')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('symmetry_regularized_clustering.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'symmetry_regularized_clustering.png'")
    plt.close()


def demo_symmetry_loss_landscape():
    """
    Demo 4: Visualize how symmetry loss changes with cluster perturbations.
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Symmetry Loss Landscape")
    print("=" * 80)
    
    # Generate a symmetric cluster
    base_cluster = generate_symmetric_cluster(
        n_points=50,
        n_dims=2,
        symmetry_type='reflection',
        noise_level=0.05,
        random_state=42
    )
    
    print("\n1. Computing symmetry loss for different perturbations...")
    
    # Create a grid of perturbations
    perturbation_strengths = np.linspace(0, 2, 20)
    symmetry_types = ['reflection', 'rotational', 'point', 'overall']
    
    results = {sym_type: [] for sym_type in symmetry_types}
    
    for strength in perturbation_strengths:
        # Add random perturbation
        perturbed = base_cluster + np.random.randn(*base_cluster.shape) * strength
        
        # Calculate different symmetry losses
        for sym_type in symmetry_types:
            loss = symmetry_loss(perturbed, symmetry_type=sym_type)
            results[sym_type].append(loss)
    
    # Visualize loss landscape
    plt.figure(figsize=(10, 6))
    
    for sym_type in symmetry_types:
        plt.plot(perturbation_strengths, results[sym_type], 
                marker='o', linewidth=2, label=f'{sym_type.capitalize()} Loss')
    
    plt.xlabel('Perturbation Strength', fontsize=12)
    plt.ylabel('Symmetry Loss', fontsize=12)
    plt.title('Symmetry Loss vs. Cluster Perturbation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('symmetry_loss_landscape.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'symmetry_loss_landscape.png'")
    plt.close()
    
    print("\n2. Loss values at different perturbation levels:")
    for i, strength in enumerate([0.0, 0.5, 1.0, 1.5, 2.0]):
        idx = int(i * 4.75)  # Approximate index
        if idx < len(perturbation_strengths):
            print(f"\nPerturbation strength: {strength:.2f}")
            for sym_type in symmetry_types:
                print(f"  {sym_type.capitalize()} Loss: {results[sym_type][idx]:.4f}")


def demo_multi_cluster_symmetry_analysis():
    """
    Demo 5: Analyze symmetry across multiple clusters simultaneously.
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Multi-Cluster Symmetry Analysis")
    print("=" * 80)
    
    # Generate multiple clusters with varying symmetry
    clusters = []
    cluster_names = []
    
    for i, (sym_type, noise) in enumerate([
        ('reflection', 0.05),
        ('rotational', 0.1),
        ('point', 0.15),
        ('reflection', 0.3)
    ]):
        cluster = generate_symmetric_cluster(
            n_points=60,
            n_dims=2,
            symmetry_type=sym_type,
            noise_level=noise,
            random_state=i
        )
        clusters.append(cluster)
        cluster_names.append(f'{sym_type.capitalize()} (noise={noise})')
    
    print("\n1. Calculating symmetry score matrix...")
    
    # Calculate symmetry scores for all clusters
    symmetry_types = ['reflection', 'rotational', 'point']
    score_matrices = {}
    
    for sym_type in symmetry_types:
        scores = calculate_symmetry_score_matrix(clusters, symmetry_type=sym_type)
        score_matrices[sym_type] = scores
        
        print(f"\n{sym_type.capitalize()} Symmetry Scores:")
        for i, name in enumerate(cluster_names):
            print(f"  {name}: {scores[i]:.4f}")
    
    # Visualize clusters with symmetry metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (cluster, name) in enumerate(zip(clusters, cluster_names)):
        ax = axes[idx]
        ax.scatter(cluster[:, 0], cluster[:, 1], alpha=0.6, s=30)
        
        # Calculate metrics
        metrics = compute_cluster_metrics(cluster)
        refl_score = calculate_reflection_symmetry(cluster, optimize=True)
        rot_score = calculate_rotational_symmetry(cluster, n_fold=2, optimize=True)
        point_score = calculate_point_symmetry(cluster)
        
        title = f"{name}\n"
        title += f"Refl: {refl_score:.3f}, Rot: {rot_score:.3f}, Point: {point_score:.3f}\n"
        title += f"Spread: {metrics['spread']:.3f}"
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('multi_cluster_symmetry.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'multi_cluster_symmetry.png'")
    plt.close()


def demo_practical_application():
    """
    Demo 6: Practical application - Using symmetry loss in optimization.
    """
    print("\n" + "=" * 80)
    print("DEMO 6: Practical Application - Symmetry-Guided Optimization")
    print("=" * 80)
    
    print("\nScenario: Optimizing point positions to maximize symmetry")
    print("This could be used in design, pattern generation, or data augmentation.\n")
    
    # Start with random points
    np.random.seed(42)
    n_points = 30
    points = np.random.randn(n_points, 2) * 2
    
    print("1. Initial configuration:")
    initial_symmetry = calculate_overall_symmetry(points)
    initial_loss = symmetry_loss(points, symmetry_type='overall')
    print(f"   Overall Symmetry: {initial_symmetry:.4f}")
    print(f"   Symmetry Loss: {initial_loss:.4f}")
    
    # Optimize for symmetry using simple gradient descent
    print("\n2. Optimizing for symmetry...")
    
    optimized_points = points.copy()
    learning_rate = 0.05
    n_iterations = 100
    
    loss_history = []
    
    for iteration in range(n_iterations):
        # Calculate current loss
        current_loss = symmetry_loss(optimized_points, symmetry_type='reflection')
        loss_history.append(current_loss)
        
        # Simple numerical gradient
        gradient = np.zeros_like(optimized_points)
        epsilon = 0.01
        
        for i in range(n_points):
            for j in range(2):
                # Perturb
                optimized_points[i, j] += epsilon
                loss_plus = symmetry_loss(optimized_points, symmetry_type='reflection')
                
                optimized_points[i, j] -= 2 * epsilon
                loss_minus = symmetry_loss(optimized_points, symmetry_type='reflection')
                
                # Restore
                optimized_points[i, j] += epsilon
                
                # Gradient
                gradient[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Update points
        optimized_points -= learning_rate * gradient
        
        if (iteration + 1) % 20 == 0:
            print(f"   Iteration {iteration + 1}: Loss = {current_loss:.4f}")
    
    print("\n3. Final configuration:")
    final_symmetry = calculate_overall_symmetry(optimized_points)
    final_loss = symmetry_loss(optimized_points, symmetry_type='overall')
    print(f"   Overall Symmetry: {final_symmetry:.4f}")
    print(f"   Symmetry Loss: {final_loss:.4f}")
    print(f"   Improvement: {(final_symmetry - initial_symmetry):.4f}")
    
    # Visualize optimization process
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Initial configuration
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=50, c='blue')
    ax.set_title(f'Initial Configuration\nSymmetry: {initial_symmetry:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Optimized configuration
    ax = axes[1]
    ax.scatter(optimized_points[:, 0], optimized_points[:, 1], alpha=0.6, s=50, c='green')
    
    # Add symmetry plane
    normal, point = find_symmetry_plane(optimized_points)
    x_range = np.array([optimized_points[:, 0].min() - 1, optimized_points[:, 0].max() + 1])
    if abs(normal[0]) > 1e-6:
        slope = -normal[1] / normal[0]
        intercept = point[1] - slope * point[0]
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, 'r--', linewidth=2, label='Symmetry Plane')
    
    ax.set_title(f'Optimized Configuration\nSymmetry: {final_symmetry:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Loss history
    ax = axes[2]
    ax.plot(loss_history, linewidth=2, color='purple')
    ax.set_title('Optimization Progress')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Symmetry Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('symmetry_optimization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'symmetry_optimization.png'")
    plt.close()


def main():
    """
    Main function to run all demos.
    """
    print("\n" + "=" * 80)
    print("SYMMETRY CALCULATION FOR POINT CLUSTERS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo shows how to calculate and use symmetry metrics for point clusters.")
    print("Symmetry can be used as a loss term in clustering algorithms to encourage")
    print("symmetric cluster formations, which is useful in many applications including:")
    print("  - Pattern recognition")
    print("  - Computer vision")
    print("  - Molecular structure analysis")
    print("  - Design optimization")
    print("  - Data augmentation")
    
    try:
        # Run all demos
        demo_basic_symmetry_calculation()
        demo_symmetry_order_estimation()
        demo_symmetry_in_clustering()
        demo_symmetry_loss_landscape()
        demo_multi_cluster_symmetry_analysis()
        demo_practical_application()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated visualizations:")
        print("  1. symmetry_clusters.png - Basic symmetry types")
        print("  2. rotational_symmetry_orders.png - Rotational symmetry estimation")
        print("  3. symmetry_regularized_clustering.png - K-means with symmetry")
        print("  4. symmetry_loss_landscape.png - Loss vs perturbation")
        print("  5. multi_cluster_symmetry.png - Multi-cluster analysis")
        print("  6. symmetry_optimization.png - Symmetry-guided optimization")
        
        print("\n" + "=" * 80)
        print("KEY TAKEAWAYS:")
        print("=" * 80)
        print("1. Symmetry scores range from 0 (no symmetry) to 1 (perfect symmetry)")
        print("2. Symmetry loss = 1 - symmetry_score, suitable for optimization")
        print("3. Different symmetry types capture different geometric properties:")
        print("   - Reflection: Mirror symmetry across a plane")
        print("   - Rotational: n-fold rotational symmetry")
        print("   - Point: Inversion symmetry through a center point")
        print("   - Overall: Combination of all symmetry types")
        print("4. Symmetry can be used as a regularization term in clustering")
        print("5. The optimize=True parameter finds the best symmetry axis/plane")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
