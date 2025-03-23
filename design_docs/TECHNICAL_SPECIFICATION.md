# Technical Specification for `alltheplots`

## 📌 Overview

`alltheplots` is a minimal, intuitive Python plotting package designed to generate instant visual insights from numerical array-like inputs. It supports arrays of various dimensionalities, with automated dimensionality detection and minimal configuration needed.

## 🚩 Features

### Public Interface:

- Single entry-point function: `alltheplots.plot(tensor: arraylike, filename: Optional[str] = None, dpi: Optional[int] = 300)`

### Supported Dimensionalities:

- **1D**: 
  - Time-domain line plot
  - Frequency-domain (FFT magnitude) plot
  - Histogram with KDE overlay (normal scale)
  - Histogram with KDE overlay (logarithmic scale)

- **2D**: *(To be defined further)*
- **3D**: *(To be defined further)*
- **4D**: *(To be defined further)*
- **nD**: *(To be defined further)*

### Supported Data Types:

- `int`
- `float`
- Planned future support: `complex`

## 📌 Dependencies:

- `seaborn`
- `matplotlib` (transitively through seaborn)
- `loguru`
- Internal reliance on `numpy` for handling arrays (no direct dependency on Torch, TensorFlow, or JAX, but compatible via numpy conversion)

## 🛠️ Compatibility:

- Compatible with all common array-likes (`numpy`, `torch`, `tensorflow`, `jax`) through internal conversions via `np.asarray()`.
- Environment auto-detection for Jupyter notebooks and terminals through internal checks (`get_ipython()` detection).

## 🎨 Customization Options:

- Saving plots directly to file with optional DPI specification.
- Users can manage plot styles externally by setting global themes in `matplotlib` or `seaborn`. This provides flexibility and adheres to best practices for managing plot aesthetics.

## 🚫 Out of Scope:

- No user-defined custom plot handlers or plugin system.
- No interactive or parameter-heavy configuration; simplicity prioritized.

## ⚙️ CI/CD Pipeline:

- Automated testing (`pytest`) for each dimensionality.
- Automated CI using GitHub Actions: formatting (`black`), linting (`flake8`), testing (`pytest`).
- Deployment via PyPI (`pip`).
-