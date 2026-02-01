"""Utility functions for symmetry analysis and visualization.

This module provides helper functions for working with point clusters,
including data generation, preprocessing, and analysis utilities.
"""

from typing import Union, Tuple, Optional, List, Dict
import numpy as np
from scipy.spatial.distance import pdist, squareform


def generate_symmetric_cluster(
    n_points: int,
    n_dims: int = 2,
    symmetry_type: str = 'reflection',
    noise_level: float = 0.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Generate a synthetic symmetric cluster of points.
    
    Useful for testing symmetry calculations and algorithms.
    
    Args:
        n_points: Number of points to generate.
        n_dims: Dimensionality of the points (2 or 3).
        symmetry_type: Type of symmetry ('reflection', 'rotational', 'point').
        noise_level: Amount of Gaussian noise to add (0 = perfect symmetry).
        random_state: Random seed for reproducibility.
    
    Returns:
        Array of shape (n_points, n_dims) with symmetric point cluster.
        
    Raises:
        ValueError: If parameters are invalid.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if n_dims not in [2, 3]:
        raise ValueError("n_dims must be 2 or 3")
    
    rng = np.random.RandomState(random_state)
    
    if symmetry_type == 'reflection':
        # Generate points on one side, then reflect
        half_n = n_points // 2
        points_half = rng.randn(half_n, n_dims)
        
        # Ensure points are on positive side of first axis
        points_half[:, 0] = np.abs(points_half[:, 0])
        
        # Reflect across first axis
        points_reflected = points_half.copy()
        points_reflected[:, 0] = -points_reflected[:, 0]
        
        points = np.vstack([points_half, points_reflected])
        
        # Add remaining point at origin if odd number
        if n_points % 2 == 1:
            origin = np.zeros((1, n_dims))
            points = np.vstack([points, origin])
    
    elif symmetry_type == 'rotational':
        # Generate points and rotate them
        base_n = n_points // 4 if n_points >= 4 else 1
        base_points = rng.randn(base_n, n_dims)
        
        points_list = [base_points]
        for i in range(1, 4):
            angle = i * np.pi / 2  # 90 degree rotations
            if n_dims == 2:
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated = base_points @ rot_matrix.T
            else:  # 3D
                # Rotate around z-axis
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                rotated = base_points @ rot_matrix.T
            points_list.append(rotated)
        
        points = np.vstack(points_list)
        
        # Trim or pad to exact n_points
        if points.shape[0] > n_points:
            points = points[:n_points]
        elif points.shape[0] < n_points:
            extra = rng.randn(n_points - points.shape[0], n_dims) * 0.1
            points = np.vstack([points, extra])
    
    elif symmetry_type == 'point':
        # Generate points and their inversions
        half_n = n_points // 2
        points_half = rng.randn(half_n, n_dims)
        points_inverted = -points_half
        
        points = np.vstack([points_half, points_inverted])
        
        # Add point at origin if odd number
        if n_points % 2 == 1:
            origin = np.zeros((1, n_dims))
            points = np.vstack([points, origin])
    
    else:
        raise ValueError(
            f"Unknown symmetry_type '{symmetry_type}'. "
            "Must be 'reflection', 'rotational', or 'point'"
        )
    
    # Add noise
    if noise_level > 0:
        noise = rng.randn(*points.shape) * noise_level
        points += noise
    
    return points


def normalize_points(
    points: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize point coordinates.
    
    Args:
        points: Array of shape (n_points, n_dims).
        method: Normalization method ('standard', 'minmax', 'center').
    
    Returns:
        Tuple of (normalized_points, normalization_params) where params
        contains information needed to reverse the normalization.
        
    Raises:
        ValueError: If method is not recognized.
    """
    points = np.asarray(points, dtype=float)
    params = {}
    
    if method == 'standard':
        # Zero mean, unit variance
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        normalized = (points - mean) / std
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        # Scale to [0, 1]
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        normalized = (points - min_vals) / range_vals
        params = {'min': min_vals, 'max': max_vals}
    
    elif method == 'center':
        # Center at origin
        mean = np.mean(points, axis=0)
        normalized = points - mean
        params = {'mean': mean}
    
    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            "Must be 'standard', 'minmax', or 'center'"
        )
    
    return normalized, params


def denormalize_points(
    points: np.ndarray,
    params: Dict[str, np.ndarray],
    method: str = 'standard'
) -> np.ndarray:
    """Reverse normalization of points.
    
    Args:
        points: Normalized points array.
        params: Normalization parameters from normalize_points().
        method: Normalization method used.
    
    Returns:
        Original-scale points.
    """
    points = np.asarray(points, dtype=float)
    
    if method == 'standard':
        return points * params['std'] + params['mean']
    elif method == 'minmax':
        range_vals = params['max'] - params['min']
        return points * range_vals + params['min']
    elif method == 'center':
        return points + params['mean']
    else:
        raise ValueError(f"Unknown normalization method '{method}'")


def compute_cluster_metrics(
    points: np.ndarray
) -> Dict[str, float]:
    """Compute various metrics for a point cluster.
    
    Args:
        points: Array of shape (n_points, n_dims).
    
    Returns:
        Dictionary containing cluster metrics:
        - 'centroid': Center of mass
        - 'spread': Average distance from centroid
        - 'compactness': Ratio of average to maximum distance
        - 'diameter': Maximum pairwise distance
        - 'density': Inverse of average pairwise distance
    """
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    
    if n_points == 0:
        return {}
    
    metrics = {}
    
    # Centroid
    centroid = np.mean(points, axis=0)
    metrics['centroid'] = centroid
    
    # Distances from centroid
    distances_from_center = np.linalg.norm(points - centroid, axis=1)
    metrics['spread'] = np.mean(distances_from_center)
    
    if n_points > 1:
        # Pairwise distances
        pairwise_dists = pdist(points)
        
        metrics['diameter'] = np.max(pairwise_dists)
        avg_pairwise = np.mean(pairwise_dists)
        metrics['density'] = 1.0 / avg_pairwise if avg_pairwise > 0 else 0.0
        
        max_dist = np.max(distances_from_center)
        metrics['compactness'] = (
            metrics['spread'] / max_dist if max_dist > 0 else 1.0
        )
    else:
        metrics['diameter'] = 0.0
        metrics['density'] = 0.0
        metrics['compactness'] = 1.0
    
    return metrics


def find_symmetry_plane(
    points: np.ndarray,
    method: str = 'pca'
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the best symmetry plane for a point cluster.
    
    Args:
        points: Array of shape (n_points, n_dims).
        method: Method to find plane ('pca' or 'optimize').
    
    Returns:
        Tuple of (normal_vector, point_on_plane).
    """
    points = np.asarray(points, dtype=float)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    if method == 'pca':
        # Use first principal component as normal
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[0]
    else:
        raise ValueError(f"Unknown method '{method}'")
    
    # Normalize
    normal = normal / np.linalg.norm(normal)
    
    return normal, centroid


def split_by_symmetry_plane(
    points: np.ndarray,
    normal: np.ndarray,
    point_on_plane: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split points into two groups by a symmetry plane.
    
    Args:
        points: Array of shape (n_points, n_dims).
        normal: Normal vector to the plane.
        point_on_plane: A point on the plane.
    
    Returns:
        Tuple of (points_positive_side, points_negative_side).
    """
    points = np.asarray(points, dtype=float)
    normal = np.asarray(normal, dtype=float)
    point_on_plane = np.asarray(point_on_plane, dtype=float)
    
    # Normalize normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signed distance from plane
    distances = np.dot(points - point_on_plane, normal)
    
    # Split points
    positive_mask = distances >= 0
    points_positive = points[positive_mask]
    points_negative = points[~positive_mask]
    
    return points_positive, points_negative


def calculate_symmetry_score_matrix(
    clusters: List[np.ndarray],
    symmetry_type: str = 'reflection'
) -> np.ndarray:
    """Calculate symmetry scores for multiple clusters.
    
    Args:
        clusters: List of point arrays, each of shape (n_points, n_dims).
        symmetry_type: Type of symmetry to measure.
    
    Returns:
        Array of symmetry scores, one per cluster.
    """
    from .core import (
        calculate_reflection_symmetry,
        calculate_rotational_symmetry,
        calculate_point_symmetry
    )
    
    symmetry_functions = {
        'reflection': calculate_reflection_symmetry,
        'rotational': calculate_rotational_symmetry,
        'point': calculate_point_symmetry
    }
    
    if symmetry_type not in symmetry_functions:
        raise ValueError(f"Unknown symmetry_type '{symmetry_type}'")
    
    func = symmetry_functions[symmetry_type]
    scores = np.array([func(cluster) for cluster in clusters])
    
    return scores


def estimate_symmetry_order(
    points: np.ndarray,
    max_order: int = 8
) -> int:
    """Estimate the order of rotational symmetry in a point cluster.
    
    Args:
        points: Array of shape (n_points, n_dims).
        max_order: Maximum symmetry order to test.
    
    Returns:
        Estimated symmetry order (2 to max_order).
    """
    from .core import calculate_rotational_symmetry
    
    best_order = 2
    best_score = float('inf')
    
    for order in range(2, max_order + 1):
        score = calculate_rotational_symmetry(points, n_fold=order)
        if score < best_score:
            best_score = score
            best_order = order
    
    return best_order
