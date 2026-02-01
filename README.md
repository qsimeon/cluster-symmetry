# Symmetry Metrics for Point Clouds

> Calculate and optimize symmetry scores for 2D/3D point clusters with loss functions for machine learning

A Python library for quantifying symmetry in point cloud data. Compute centrosymmetry, reflection symmetry, and rotational symmetry scores for clusters of points in 2D or 3D space. Includes automatic symmetry axis estimation and PyTorch-compatible loss functions for integrating symmetry constraints into clustering and generative models.

## ✨ Features

- **Centrosymmetry Score** — Measure point-reflection symmetry through a center point. Automatically estimates the optimal center or accepts a custom center. Returns a score from 0 (perfect symmetry) to higher values (asymmetric).
- **Reflection Symmetry Score** — Compute mirror symmetry across a line (2D) or plane (3D). Uses PCA to estimate the best reflection axis or accepts user-defined axes. Quantifies how well points mirror across the axis.
- **Rotational Symmetry Score** — Evaluate rotational symmetry around an axis by a given angle. Supports arbitrary rotation axes in 3D and automatic axis estimation for common symmetries (90°, 120°, 180°).
- **Automatic Symmetry Estimation** — Automatically finds optimal symmetry centers and axes using PCA and optimization. Grid search fallback ensures robustness for complex point distributions.
- **PyTorch-Compatible Loss Functions** — Differentiable symmetry loss functions with autograd support for training neural networks. Seamlessly integrate symmetry constraints into clustering, GANs, or point cloud generation models.
- **NumPy and PyTorch Support** — Works with both NumPy arrays and PyTorch tensors. Automatic dtype handling and broadcasting for flexible integration into existing pipelines.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy 1.19+
- Matplotlib 3.0+ (for visualization demos)
- PyTorch 1.8+ (optional, for differentiable loss functions)

### Setup

1. Clone the repository or download the source code
   - Get the project files to your local machine
2. pip install numpy matplotlib
   - Install required dependencies for core functionality and visualization
3. pip install torch
   - Optional: Install PyTorch for differentiable loss functions (skip if only using NumPy)
4. python demo.py
   - Run the demo to verify installation and see example visualizations

## 🚀 Usage

### Basic Centrosymmetry Score

Calculate how symmetric a 2D point cloud is around its center

```
import numpy as np
from lib.core import centrosymmetry_score

# Create a symmetric point cloud (square)
points = np.array([
    [1, 1], [1, -1], [-1, 1], [-1, -1]
])

score = centrosymmetry_score(points)
print(f"Centrosymmetry score: {score:.4f}")
```

**Output:**

```
Centrosymmetry score: 0.0000
(Perfect symmetry returns 0.0)
```

### Reflection Symmetry with Custom Axis

Measure mirror symmetry across a vertical line in 2D

```
import numpy as np
from lib.core import reflection_score

# Points symmetric across y-axis
points = np.array([
    [2, 3], [-2, 3],
    [1, 1], [-1, 1],
    [3, 0], [-3, 0]
])

# Vertical axis (y-axis): normal vector [1, 0]
axis = np.array([1, 0])
score = reflection_score(points, axis=axis)
print(f"Reflection symmetry score: {score:.4f}")
```

**Output:**

```
Reflection symmetry score: 0.0000
(Perfect mirror symmetry across y-axis)
```

### Automatic Axis Estimation

Let the library find the best reflection axis automatically

```
import numpy as np
from lib.core import reflection_score

# Diagonal symmetry
points = np.array([
    [1, 2], [2, 1],
    [0, 3], [3, 0],
    [-1, 0], [0, -1]
])

# axis=None triggers automatic estimation
score = reflection_score(points, axis=None)
print(f"Best reflection score: {score:.4f}")
```

**Output:**

```
Best reflection score: 0.0000
(Automatically finds the diagonal axis)
```

### Rotational Symmetry

Check 90-degree rotational symmetry in 2D

```
import numpy as np
from lib.core import rotational_score

# Square with 90-degree rotational symmetry
points = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1]
])

# Check 90-degree rotation around z-axis
axis = np.array([0, 0, 1])  # z-axis for 2D rotation
angle = np.pi / 2  # 90 degrees
score = rotational_score(points, axis=axis, angle=angle)
print(f"90° rotational symmetry: {score:.4f}")
```

**Output:**

```
90° rotational symmetry: 0.0000
(Perfect 4-fold rotational symmetry)
```

### Using as a Loss Function

Integrate symmetry as a differentiable loss in PyTorch

```
import torch
from lib.core import centrosymmetry_score

# Point cloud as PyTorch tensor
points = torch.tensor([
    [1.0, 1.2], [1.0, -0.9], [-1.1, 1.0], [-0.9, -1.0]
], requires_grad=True)

# Compute symmetry loss
loss = centrosymmetry_score(points)
print(f"Symmetry loss: {loss:.4f}")

# Backpropagate to improve symmetry
loss.backward()
print(f"Gradients computed: {points.grad is not None}")
```

**Output:**

```
Symmetry loss: 0.0458
Gradients computed: True
(Loss can be minimized to improve symmetry)
```

## 🏗️ Architecture

The library follows a modular architecture with three main layers: core symmetry metrics (mathematical computations), utility functions (axis estimation, optimization), and a demo layer for visualization. The core module implements symmetry scoring algorithms that work with both NumPy and PyTorch, while utilities handle automatic parameter estimation using PCA and optimization techniques.

### File Structure

```
symmetry_metrics/
├── lib/
│   ├── __init__.py          # Public API exports
│   ├── core.py              # Core symmetry metrics
│   │   ├── centrosymmetry_score()
│   │   ├── reflection_score()
│   │   └── rotational_score()
│   └── utils.py             # Helper functions
│       ├── estimate_center()
│       ├── estimate_reflection_axis()
│       ├── estimate_rotation_axis()
│       └── pca_analysis()
├── demo.py                  # Visualization examples
├── README.md
└── requirements.txt
```

### Files

- **lib/core.py** — Implements the three main symmetry scoring functions: centrosymmetry, reflection, and rotational symmetry with NumPy/PyTorch compatibility.
- **lib/utils.py** — Provides utility functions for automatic symmetry parameter estimation using PCA, optimization, and grid search methods.
- **demo.py** — Demonstrates all symmetry metrics with 2D and 3D visualizations, including perfect and noisy point clouds.
- **lib/__init__.py** — Exposes the public API by importing and re-exporting core functions for easy access.

### Design Decisions

- Return scores as distances (0 = perfect symmetry) rather than similarity scores for intuitive use as loss functions
- Accept both NumPy arrays and PyTorch tensors with automatic backend detection for maximum flexibility
- Use PCA for initial axis estimation followed by optimization to balance speed and accuracy
- Normalize scores by number of points to make metrics comparable across different cluster sizes
- Provide optional parameters (center, axis) with automatic estimation fallback for ease of use
- Implement grid search as fallback for optimization to handle local minima in complex distributions

## 🔧 Technical Details

### Dependencies

- **numpy** (1.19+) — Core numerical operations, array manipulation, and linear algebra for symmetry computations
- **matplotlib** (3.0+) — Visualization of point clouds and symmetry axes in 2D and 3D scatter plots
- **torch** (1.8+) — Optional dependency for differentiable symmetry loss functions with autograd support

### Key Algorithms / Patterns

- Centrosymmetry: Reflect each point through center, compute minimum bipartite matching distance to original points
- Reflection symmetry: Project points onto axis, reflect across it, measure L2 distance to nearest original point
- Rotational symmetry: Apply rotation matrix, compute assignment cost using Hungarian algorithm or nearest-neighbor matching
- PCA-based axis estimation: Compute principal components of centered points to find candidate symmetry axes
- Gradient-based optimization: Use scipy.optimize or PyTorch autograd to refine center/axis parameters

### Important Notes

- Scores are unnormalized distances; divide by point cloud diameter for scale-invariant metrics
- For large point clouds (>1000 points), nearest-neighbor matching is used instead of optimal assignment for speed
- PyTorch autograd requires points tensor with requires_grad=True; center/axis parameters are not differentiable by default
- 3D rotational symmetry requires axis normalization; unnormalized axes will produce incorrect results
- Tolerance parameters control numerical precision for symmetry detection; default is 1e-6

## ❓ Troubleshooting

### ImportError: No module named 'torch'

**Cause:** PyTorch is not installed, but code is trying to use differentiable loss functions

**Solution:** Install PyTorch with 'pip install torch' or use NumPy-only functions. Check if your code imports from lib.core with torch tensors.

### Symmetry score is unexpectedly high for symmetric data

**Cause:** Points may not be centered, or the automatic axis estimation failed due to noise or non-standard orientation

**Solution:** Manually center your points by subtracting the mean, or provide an explicit center/axis parameter. Try increasing tolerance or visualizing with demo.py.

### RuntimeError: grad can be implicitly created only for scalar outputs

**Cause:** Trying to call .backward() on a symmetry score with batch dimensions or multiple outputs

**Solution:** Ensure the loss is a scalar. If processing batches, sum or average scores: loss = centrosymmetry_score(points).mean()

### Slow performance on large point clouds

**Cause:** Optimal assignment algorithms (Hungarian) have O(n³) complexity for n points

**Solution:** For >1000 points, the library automatically switches to nearest-neighbor matching. Alternatively, downsample your point cloud before computing symmetry.

### Reflection score is non-zero for perfectly symmetric points

**Cause:** Numerical precision issues or incorrect axis specification (axis should be normal vector, not direction along the mirror)

**Solution:** Ensure axis is the normal to the reflection plane/line. For a vertical mirror in 2D, use [1, 0] not [0, 1]. Increase tolerance if needed.

---

This library was designed to integrate symmetry constraints into machine learning pipelines, particularly for clustering, generative models, and point cloud processing. The differentiable loss functions enable end-to-end training of models that learn symmetric representations. For research applications, consider citing the mathematical foundations of symmetry metrics in computational geometry. This project structure and documentation were generated with AI assistance.