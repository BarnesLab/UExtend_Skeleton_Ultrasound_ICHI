# U-Extend Skeleton + Ultrasound (ICHI) Pipeline

## Purpose

This repository contains the experimental pipeline used for multimodal analysis of upper-limb motor function using synchronized 3D skeleton trajectories and ultrasound biomarkers. The code supports preprocessing, alignment in shape space, feature extraction, statistical modeling, and classification experiments associated with the ICHI study.

The workflow combines:

* Pose trajectories (2D and 3D joint coordinates)
* Ultrasound derived signals
* Time normalization and segmentation
* Riemannian alignment and mean shape estimation
* Feature extraction
* Statistical modeling and classification

The repository is notebook-centric and is intended primarily for research reproducibility rather than packaging as a standalone library.

---

## Repository Structure

| File                                         | Role                                     |
| -------------------------------------------- | ---------------------------------------- |
| `2D_pose.ipynb`                              | Processing raw 2D pose detections        |
| `multi_pose3D.ipynb`                         | 3D skeleton construction and cleaning    |
| `segmentation.ipynb`                         | Motion segmentation into task phases     |
| `normalize time.ipynb`                       | Temporal alignment / resampling          |
| `alignment_functions.py`                     | Core geometric alignment operations      |
| `functionsgpu.py`                            | GPU-accelerated curve operations         |
| `functionsjoint.py`                          | Joint trajectory geometry utilities      |
| `functionsjointgpu.py`                       | GPU version of joint utilities           |
| `Skeleton_Features.ipynb`                    | Skeleton feature extraction              |
| `US_data_Process.ipynb`                      | Ultrasound preprocessing                 |
| `feature_extractor.ipynb`                    | Multimodal feature assembly              |
| `f_mean_all_pca.ipynb`                       | Mean shape + PCA modeling                |
| `fmean_left_right.ipynb`                     | Left/right limb statistics               |
| `classification.ipynb`                       | Final ML classification experiments      |
| `shap_example.ipynb`                         | Model interpretability                   |
| `plotting_beta.py`, `plotting_betas.py`      | Visualization of coefficients            |
| `brook_stat.ipynb`                           | Statistical testing                      |
| `labelfile_creator.ipynb`                    | Label preparation                        |
| `Action_separate.ipynb`                      | Action splitting                         |
| `Renaming.ipynb`                             | Dataset cleanup utilities                |
| `demo.R`, `mfpca.face.R`, `face.Cov.mfpca.R` | Functional PCA statistical analysis in R |

---

## Expected Data Organization

The notebooks assume a dataset layout resembling:

```
data/
  subject_01/
    pose2d/
    pose3d/
    ultrasound/
    labels.csv
  subject_02/
    ...
```

Each trial should contain synchronized frames across modalities.
Sampling frequency consistency is assumed before time normalization.

---

## Environment Setup

No environment file is provided. A minimal working environment typically requires:

### Python

```
python >= 3.9
numpy
scipy
pandas
matplotlib
scikit-learn
jupyter
notebook
seaborn
tqdm
numba
torch
shap
```

Optional (for GPU acceleration):

```
cuda-enabled pytorch
cupy (optional depending on GPU utilities usage)
```

Create environment example:

```
conda create -n uextend python=3.10
conda activate uextend
pip install numpy scipy pandas matplotlib scikit-learn jupyter seaborn tqdm numba torch shap
```

### R

Required for functional PCA notebooks:

```
R >= 4.0
refund
fdapace
fda
MFPCA
```

---

## Recommended Execution Order

The pipeline is sequential and stateful. Run notebooks in this order.

### Stage 1 Pose Construction

1. `2D_pose.ipynb`
2. `multi_pose3D.ipynb`

### Stage 2 Cleaning & Segmentation

3. `segmentation.ipynb`
4. `normalize time.ipynb`

### Stage 3 Alignment & Geometry

5. `alignment_functions.py` (used internally)
6. `Skeleton_Features.ipynb`

### Stage 4 Ultrasound Processing

7. `US_data_Process.ipynb`

### Stage 5 Feature Construction

8. `feature_extractor.ipynb`

### Stage 6 Statistical Modeling

9. `f_mean_all_pca.ipynb`
10. `fmean_left_right.ipynb`
11. R scripts (`mfpca.face.R`, `face.Cov.mfpca.R`)

### Stage 7 Machine Learning

12. `classification.ipynb`
13. `shap_example.ipynb`

---

## Conceptual Pipeline

1. Convert pose detections ? continuous trajectories
2. Segment functional movement phases
3. Time normalize curves
4. Align curves in shape space
5. Compute mean shapes and PCA components
6. Combine ultrasound biomarkers
7. Train classification models
8. Interpret using SHAP

---

## GPU Acceleration

If CUDA is available, `functionsgpu.py` and `functionsjointgpu.py` provide faster curve operations for large cohorts. CPU versions remain available for reproducibility.

---

## Reproducibility Notes

* Notebooks assume manual path editing
* Intermediate outputs are reused across later notebooks
* Clearing outputs will require rerunning earlier stages
* Random seeds are not globally fixed in all notebooks

---

## Citation

If using this repository in academic work, cite the associated ICHI publication describing the multimodal skeleton-ultrasound analysis framework.

---

## Intended Audience

This repository is designed for researchers familiar with:

* Functional data analysis
* Riemannian shape alignment
* Biomechanics trajectory analysis
* Multimodal medical ML pipelines

It is not a packaged software library but a research workflow.

