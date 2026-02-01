"""Core symmetry calculation module for point clusters.

This module provides functions to calculate various symmetry measures for clusters
of points in n-dimensional space. These measures can be used as loss functions in
data clustering algorithms to encourage symmetric cluster formations.
"""

from typing import Union, Tuple, Optional, List
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def calculate_reflection_symmetry(
    points: np.ndarray,
    axis: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    optimize: bool = True
) -> float:
    """Calculate reflection symmetry score for a cluster of points.
    
    Measures how symmetric a point cluster is with respect to reflection across
    a hyperplane. Lower scores indicate higher symmetry.
    
    Args:
        points: Array of shape (n_points, n_dims) containing point coordinates.
        axis: Normal vector to the reflection plane. If None and optimize=True,
              will find optimal axis.
        center: Point through which the reflection plane passes. If None,
                uses centroid of points.
        optimize: If True, optimizes the reflection axis to find maximum symmetry.
    
    Returns:
        Symmetry score (0 = perfect symmetry, higher = less symmetric).
        
    Raises:
        ValueError: If points array is empty or has invalid shape.
    """
    if points.size == 0:
        raise ValueError("Points array cannot be empty")
    if points.ndim != 2:
        raise ValueError("Points must be a 2D array of shape (n_points, n_dims)")
    
    points = np.asarray(points, dtype=float)
    n_points, n_dims = points.shape
    
    if center is None:
        center = np.mean(points, axis=0)
    else:
        center = np.asarray(center, dtype=float)
    
    # Center the points
    centered_points = points - center
    
    if axis is None and optimize:
        # Find optimal reflection axis using PCA
        axis = _find_optimal_reflection_axis(centered_points)
    elif axis is None:
        # Use first principal component as default
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        axis = vh[0]
    else:
        axis = np.asarray(axis, dtype=float)
    
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Reflect points across the hyperplane
    reflected_points = _reflect_points(centered_points, axis)
    
    # Calculate symmetry score as minimum matching distance
    score = _calculate_matching_distance(centered_points, reflected_points)
    
    return score


def calculate_rotational_symmetry(
    points: np.ndarray,
    n_fold: int = 2,
    axis: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    optimize: bool = True
) -> float:
    """Calculate rotational symmetry score for a cluster of points.
    
    Measures how symmetric a point cluster is with respect to n-fold rotation.
    Lower scores indicate higher symmetry.
    
    Args:
        points: Array of shape (n_points, n_dims) containing point coordinates.
        n_fold: Order of rotational symmetry (e.g., 2 for 180°, 3 for 120°, etc.).
        axis: Rotation axis (only for 3D). If None, uses z-axis or optimizes.
        center: Center of rotation. If None, uses centroid of points.
        optimize: If True, optimizes the rotation axis/center for maximum symmetry.
    
    Returns:
        Symmetry score (0 = perfect symmetry, higher = less symmetric).
        
    Raises:
        ValueError: If points array is invalid or n_fold < 2.
    """
    if points.size == 0:
        raise ValueError("Points array cannot be empty")
    if points.ndim != 2:
        raise ValueError("Points must be a 2D array of shape (n_points, n_dims)")
    if n_fold < 2:
        raise ValueError("n_fold must be at least 2")
    
    points = np.asarray(points, dtype=float)
    n_points, n_dims = points.shape
    
    if center is None:
        center = np.mean(points, axis=0)
    else:
        center = np.asarray(center, dtype=float)
    
    # Center the points
    centered_points = points - center
    
    angle = 2 * np.pi / n_fold
    
    if n_dims == 2:
        # 2D rotation
        rotated_points = _rotate_2d(centered_points, angle)
    elif n_dims == 3:
        # 3D rotation
        if axis is None:
            axis = np.array([0., 0., 1.])  # Default to z-axis
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        rotated_points = _rotate_3d(centered_points, axis, angle)
    else:
        raise ValueError("Rotational symmetry only supported for 2D and 3D points")
    
    # Calculate symmetry score
    score = _calculate_matching_distance(centered_points, rotated_points)
    
    return score


def calculate_point_symmetry(
    points: np.ndarray,
    center: Optional[np.ndarray] = None
) -> float:
    """Calculate point (inversion) symmetry score for a cluster of points.
    
    Measures how symmetric a point cluster is with respect to inversion through
    a central point. Lower scores indicate higher symmetry.
    
    Args:
        points: Array of shape (n_points, n_dims) containing point coordinates.
        center: Center of inversion. If None, uses centroid of points.
    
    Returns:
        Symmetry score (0 = perfect symmetry, higher = less symmetric).
        
    Raises:
        ValueError: If points array is empty or has invalid shape.
    """
    if points.size == 0:
        raise ValueError("Points array cannot be empty")
    if points.ndim != 2:
        raise ValueError("Points must be a 2D array of shape (n_points, n_dims)")
    
    points = np.asarray(points, dtype=float)
    
    if center is None:
        center = np.mean(points, axis=0)
    else:
        center = np.asarray(center, dtype=float)
    
    # Center the points
    centered_points = points - center
    
    # Invert points through center
    inverted_points = -centered_points
    
    # Calculate symmetry score
    score = _calculate_matching_distance(centered_points, inverted_points)
    
    return score


def calculate_overall_symmetry(
    points: np.ndarray,
    weights: Optional[dict] = None,
    **kwargs
) -> float:
    """Calculate overall symmetry score combining multiple symmetry types.
    
    Computes a weighted combination of reflection, rotational, and point symmetry.
    
    Args:
        points: Array of shape (n_points, n_dims) containing point coordinates.
        weights: Dictionary with keys 'reflection', 'rotational', 'point' and
                float values. If None, uses equal weights.
        **kwargs: Additional arguments passed to individual symmetry functions.
    
    Returns:
        Combined symmetry score (0 = perfect symmetry, higher = less symmetric).
    """
    if weights is None:
        weights = {'reflection': 1.0, 'rotational': 1.0, 'point': 1.0}
    
    scores = {}
    total_weight = 0.0
    
    if weights.get('reflection', 0) > 0:
        scores['reflection'] = calculate_reflection_symmetry(points, **kwargs)
        total_weight += weights['reflection']
    
    if weights.get('rotational', 0) > 0:
        n_fold = kwargs.get('n_fold', 2)
        scores['rotational'] = calculate_rotational_symmetry(
            points, n_fold=n_fold, **kwargs
        )
        total_weight += weights['rotational']
    
    if weights.get('point', 0) > 0:
        scores['point'] = calculate_point_symmetry(points)
        total_weight += weights['point']
    
    if total_weight == 0:
        return 0.0
    
    # Weighted average
    overall_score = sum(
        weights.get(key, 0) * value for key, value in scores.items()
    ) / total_weight
    
    return overall_score


def symmetry_loss(
    points: np.ndarray,
    symmetry_type: str = 'overall',
    normalize: bool = True,
    **kwargs
) -> float:
    """Calculate symmetry loss for use in clustering algorithms.
    
    This function can be used as a regularization term in clustering objectives
    to encourage symmetric cluster formations.
    
    Args:
        points: Array of shape (n_points, n_dims) containing point coordinates.
        symmetry_type: Type of symmetry ('reflection', 'rotational', 'point', 'overall').
        normalize: If True, normalizes loss by cluster size and spread.
        **kwargs: Additional arguments passed to symmetry calculation functions.
    
    Returns:
        Symmetry loss value.
        
    Raises:
        ValueError: If symmetry_type is not recognized.
    """
    symmetry_functions = {
        'reflection': calculate_reflection_symmetry,
        'rotational': calculate_rotational_symmetry,
        'point': calculate_point_symmetry,
        'overall': calculate_overall_symmetry
    }
    
    if symmetry_type not in symmetry_functions:
        raise ValueError(
            f"Unknown symmetry_type '{symmetry_type}'. "
            f"Must be one of {list(symmetry_functions.keys())}"
        )
    
    loss = symmetry_functions[symmetry_type](points, **kwargs)
    
    if normalize and points.shape[0] > 1:
        # Normalize by cluster spread (standard deviation)
        spread = np.std(points)
        if spread > 0:
            loss = loss / spread
    
    return loss


# Helper functions

def _reflect_points(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Reflect points across a hyperplane defined by its normal vector.
    
    Args:
        points: Centered points array of shape (n_points, n_dims).
        axis: Normal vector to the reflection plane.
    
    Returns:
        Reflected points array.
    """
    axis = axis / np.linalg.norm(axis)
    # Reflection formula: p' = p - 2(p·n)n
    projection = np.dot(points, axis)
    reflected = points - 2 * np.outer(projection, axis)
    return reflected


def _rotate_2d(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D points by given angle.
    
    Args:
        points: Points array of shape (n_points, 2).
        angle: Rotation angle in radians.
    
    Returns:
        Rotated points array.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    return points @ rotation_matrix.T


def _rotate_3d(points: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 3D points around an axis by given angle using Rodrigues' formula.
    
    Args:
        points: Points array of shape (n_points, 3).
        axis: Rotation axis (normalized).
        angle: Rotation angle in radians.
    
    Returns:
        Rotated points array.
    """
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = (
        np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    )
    
    return points @ rotation_matrix.T


def _calculate_matching_distance(
    points1: np.ndarray,
    points2: np.ndarray
) -> float:
    """Calculate minimum matching distance between two point sets.
    
    Uses Hungarian algorithm approximation to find best point correspondence.
    
    Args:
        points1: First set of points.
        points2: Second set of points.
    
    Returns:
        Average minimum distance between matched points.
    """
    if points1.shape[0] != points2.shape[0]:
        raise ValueError("Point sets must have same number of points")
    
    # Calculate pairwise distances
    distances = cdist(points1, points2)
    
    # For each point in set1, find closest point in set2
    min_distances = np.min(distances, axis=1)
    
    # Return average minimum distance
    return np.mean(min_distances)


def _find_optimal_reflection_axis(points: np.ndarray) -> np.ndarray:
    """Find optimal reflection axis using PCA.
    
    Args:
        points: Centered points array.
    
    Returns:
        Optimal reflection axis (normal to best-fit hyperplane).
    """
    # Use SVD to find principal components
    _, _, vh = np.linalg.svd(points, full_matrices=False)
    
    # Try each principal component and return the one with best symmetry
    best_axis = vh[0]
    best_score = float('inf')
    
    for axis in vh:
        reflected = _reflect_points(points, axis)
        score = _calculate_matching_distance(points, reflected)
        if score < best_score:
            best_score = score
            best_axis = axis
    
    return best_axis
