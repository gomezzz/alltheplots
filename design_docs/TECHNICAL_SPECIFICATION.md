# Technical Specification for `alltheplots`

## 📌 Overview

`alltheplots` is a minimal, intuitive Python plotting package designed to generate instant visual insights from numerical array-like inputs. It supports arrays of various dimensionalities, with automated dimensionality detection and minimal configuration needed.

## 🚩 Features

### Public Interface:

- Single entry-point function: `alltheplots.plot(tensor: arraylike, filename: Optional[str] = None, dpi: int = 100, show: bool = True)`

### Supported Dimensionalities:

- **1D (Time Series, Signals, etc.):**

| Domain                 | Row 1             | Row 2                | Row 3                  |
| ---------------------- | ----------------- | -------------------- | ---------------------- |
| **Time Domain**        | Line/scatter plot | Line/scatter (log-x) | Line/scatter (log-y)   |
| **Frequency Domain**   | FFT Magnitude     | Autocorrelation      | Power Spectral Density |
| **Value Distribution** | Histogram + KDE   | Violin plot          | CDF                    |

- **2D (Images, Matrices, etc.):**

| Domain             | Row 1              | Row 2             | Row 3                |
| ------------------ | ------------------ | ----------------- | -------------------- |
| **3D Views**       | 3D Surface (Front) | 3D Surface (Side) | 3D Surface (Top)     |
| **Distribution**   | Histogram + KDE    | Row means profile | Column means profile |
| **Shape Analysis** | Contour plot       | 2D FFT Magnitude  | Heatmap              |

- **3D (Volumes, Stacks, etc.):**

| Domain                | Row 1                   | Row 2                   | Row 3                   |
| --------------------- | ----------------------- | ----------------------- | ----------------------- |
| **Slice Views**       | Central XY slice        | Central XZ slice        | Central YZ slice        |
| **3D Visualizations** | 3D surface (X-axis avg) | 3D surface (Y-axis avg) | 3D surface (Z-axis avg) |
| **Distribution**      | 2D projection (t-SNE)   | Histogram + KDE         | CDF                     |

- **nD (Higher Dimensions):**

| Domain                    | Row 1           | Row 2                   | Row 3           |
| ------------------------- | --------------- | ----------------------- | --------------- |
| **Dimension Reduction**   | PCA projection  | t-SNE projection        | UMAP projection |
| **Aggregate Projections** | Mean projection | Standard dev projection | Max projection  |
| **Value Distribution**    | Histogram       | KDE                     | CDF             |

These layouts can evolve to best support specific use cases (e.g., scientific images, sensor grids, spatiotemporal data).

### Supported Data Types:

- `int`
- `float`

## 📌 Dependencies:

- `matplotlib` - Core plotting library
- `numpy` - Array manipulation
- `scipy` - Scientific computing
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning for dimensionality reduction
- `umap-learn` - UMAP dimensionality reduction for high-dimensional data
- `loguru` - Logging

## 🛠️ Compatibility:

- Compatible with all common array-likes (`numpy`, `torch`, `tensorflow`, `jax`) through internal conversions via `to_numpy()` utility.
- Environment auto-detection for Jupyter notebooks and terminals.

## 🎨 Customization Options:

- Saving plots directly to file with optional DPI specification.
- Users can manage plot styles externally by setting global themes in `matplotlib` or `seaborn`. This provides flexibility and adheres to best practices for managing plot aesthetics.

## 🚫 Out of Scope:

- No user-defined custom plot handlers or plugin system.
- No interactive or parameter-heavy configuration; simplicity prioritized.

## ⚙️ CI/CD Pipeline:

- Automated testing (`pytest`) for each dimensionality.
- Automated CI using GitHub Actions: formatting (`black`), linting (`flake8`), testing (`pytest`).
- Deployment via PyPI.