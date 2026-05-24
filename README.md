# Image Segmentation Unified Project

## Overview

This project merges **4 independent image processing scenes** into a single, unified Python program. Each scene performs image segmentation using different techniques and is designed to work both **independently and together**.

### Scenes Included

| Scene | Task | Method | Details |
|-------|------|--------|---------|
| **1** | Chat/Ciel/Sol/Arbres | K-Means (k=4) | RGB→LAB + Gaussian blur + clustering |
| **2** | Retinal Bright Disk | Otsu Threshold + Morphology | Green channel + CLAHE + top-hat |
| **3** | Aerial Roads | K-Means (k=3) | RGB→LAB + Gaussian blur + road cluster detection |
| **4** | Person Extraction | K-Means Multi-channel (k=3) | RGB + Depth features [R,G,B,D] + center-aware clustering |

---

## Project Structure

```
TAI FD/
├── image_segmentation_unified.py  # Main merged program
├── CLAUDE.md                      # Developer documentation
├── README.md                      # This file
├── Scene_1.png, GT1.png          # Scene 1 images
├── Scene_2.png, GT2.png          # Scene 2 images
├── Scene_3.png, GT3.png          # Scene 3 images
├── Scene_4_RGB_1.png             # Scene 4 RGB image
├── Scene_4_D_2.png               # Scene 4 Depth image
├── GT4.png                       # Scene 4 Ground truth
├── resultats_scene1/             # Scene 1 output directory
├── resultats_scene2/             # Scene 2 output directory
├── resultats_scene3/             # Scene 3 output directory
└── resultats_scene4/             # Scene 4 output directory
```

---

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install required packages:

```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image
```

Or install all at once:

```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image
```

---

## Usage

### Run All Scenes

```bash
python image_segmentation_unified.py all
```

or simply:

```bash
python image_segmentation_unified.py
```

### Run Specific Scene(s)

```bash
# Run scene 1 only
python image_segmentation_unified.py 1

# Run scenes 1 and 3
python image_segmentation_unified.py 1 3

# Run scenes 1, 2, 3, and 4
python image_segmentation_unified.py 1 2 3 4
```

---

## Program Output

Each scene generates:

1. **Visualization PNG** - Side-by-side comparison of:
   - Original image
   - Preprocessing steps
   - Segmented result
   - Ground truth mask
   - Metrics table (if applicable)

2. **Console Metrics** - Printed results:
   - IoU, Dice, Precision, Recall
   - Cluster identification (Scene 1)
   - Component analysis (Scene 2, 3, 4)

3. **Output Directories**:
   - `resultats_scene1/` → scene1_results.png
   - `resultats_scene2/` → scene2_results.png
   - `resultats_scene3/` → scene3_results.png
   - `resultats_scene4/` → scene4_results.png

---

## Key Changes Made During Merge

### 1. **Code Consolidation**
   - ✅ Eliminated duplicate imports (cv2, numpy, matplotlib)
   - ✅ Removed conflicting function names
   - ✅ Unified metrics computation into `MetricsComputer` class

### 2. **Object-Oriented Design**
   - ✅ Each scene is now a class (`Scene1`, `Scene2`, `Scene3`, `Scene4`)
   - ✅ Each class has standardized methods: `load_images()`, `preprocess()`, `segment_*()`, `compute_metrics()`, `visualize()`, `run()`
   - ✅ Main orchestrator: `ImageSegmentationPipeline`

### 3. **Shared Evaluation Module**
   - ✅ `MetricsComputer.compute_metrics_basic()` - For scenes 1, 2, 3
   - ✅ `MetricsComputer.compute_metrics_extended()` - For scene 4
   - ✅ `MetricsComputer.print_metrics()` - Unified output formatting

### 4. **File Path Handling**
   - ✅ Parameterized image paths instead of hard-coded paths
   - ✅ Output directories automatically created
   - ✅ Flexible scene selection via command-line arguments

### 5. **Algorithm Preservation**
   - ✅ All original preprocessing logic preserved
   - ✅ All clustering and thresholding methods unchanged
   - ✅ Metrics calculation identical to original scripts
   - ✅ No loss of functionality

### 6. **Error Handling**
   - ✅ Proper file loading error messages
   - ✅ Invalid scene number handling
   - ✅ GT availability checks before metric computation

---

## Implementation Details

### Metrics Computation

All scenes use the same metrics:

```python
TP = True Positives    (pixels correctly identified as target)
FP = False Positives   (pixels incorrectly identified as target)
FN = False Negatives   (target pixels missed)
TN = True Negatives    (background pixels correctly identified)

IoU        = TP / (TP + FP + FN)
Dice       = 2*TP / (2*TP + FP + FN)
Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
Accuracy   = (TP + TN) / (TP + TN + FP + FN)  [Scene 4 only]
```

### Scene-Specific Pipeline Highlights

**Scene 1 (K-Means k=4)**:
- RGB to LAB conversion for better color separation
- MinMax normalization for stable clustering
- Cluster identification by matching GT colors

**Scene 2 (Otsu + Morphology)**:
- Green channel extraction (most relevant for retinal imaging)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Top-hat morphology to isolate bright structures
- Circular shape filtering for disk detection

**Scene 3 (K-Means k=3)**:
- Similar to Scene 1 but with fewer clusters (k=3)
- Road cluster identified by maximum L-channel value (brightest)
- Red color extraction from GT for roads

**Scene 4 (Multi-channel K-Means k=3)**:
- Novel 4-channel feature matrix: [R, G, B, Depth]
- Hybrid cluster identification: 70% center position + 30% depth value
- Connected components filtering for person silhouette

---

## Verification Checklist

### ✅ Code Integration

- [x] All imports consolidated without conflicts
- [x] No function name collisions
- [x] Metrics computed identically to originals
- [x] File paths parameterized
- [x] Output directories auto-created

### ✅ Scene 1 Validation

- [x] Images load correctly
- [x] Preprocessing produces expected output
- [x] K-Means converges with k=4
- [x] 4 clusters identified (chat, arbres, sol, ciel)
- [x] Metrics display for each cluster
- [x] Visualization shows all steps
- [x] Results match original seg.py output

### ✅ Scene 2 Validation

- [x] Images load correctly
- [x] Green channel extracted properly
- [x] CLAHE enhancement applied
- [x] Otsu threshold works
- [x] Morphological operations clean noise
- [x] Bright disk extracted successfully
- [x] Metrics computed correctly
- [x] Results match original projet_tai_fd_scene2.py output

### ✅ Scene 3 Validation

- [x] Images load correctly
- [x] Preprocessing produces expected output
- [x] K-Means converges with k=3
- [x] Road cluster identified (brightest)
- [x] Morphological operations refine mask
- [x] GT roads extracted by color (red channel)
- [x] Metrics display correctly
- [x] Results match original scene3_routes.py output

### ✅ Scene 4 Validation

- [x] Color and depth images load correctly
- [x] Images resized to matching dimensions
- [x] Feature matrix built correctly [R,G,B,D]
- [x] K-Means converges with k=3
- [x] Person cluster identified using hybrid strategy
- [x] Morphological operations clean silhouette
- [x] Connected components filtering works
- [x] Metrics computed with extended set
- [x] Results match original mapartie.py output

### ✅ Unified Program Validation

- [x] Command-line arguments parsed correctly
- [x] All scenes run independently
- [x] All scenes run together
- [x] Output directories created automatically
- [x] Visualizations display without errors
- [x] Metrics print clearly to console
- [x] Error handling works for missing files
- [x] No hard dependencies on Colab or specific paths

### ✅ Output Quality

- [x] Visualization PNGs save correctly
- [x] Console output is clear and informative
- [x] Metrics values are reasonable
- [x] No warnings or errors during execution
- [x] Processing time acceptable

---

## Troubleshooting

### Issue: "Image not found"
**Solution**: Ensure Scene_*.png and GT*.png files exist in the working directory.

### Issue: "Module not found (cv2, numpy, etc.)"
**Solution**: Run `pip install opencv-python numpy matplotlib scikit-learn scikit-image`

### Issue: "Specific scene won't run"
**Solution**: Check console error message, ensure input images are valid PNG files with correct dimensions.

### Issue: "Metrics look wrong"
**Solution**: Verify ground truth masks are properly loaded and have correct shape. Check console output for shape mismatches.

### Issue: "Matplotlib not displaying"
**Solution**: Depends on your environment (Jupyter, IDE, terminal). Try adding before running:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system
```

---

## What Changed and Why

### Original Problems:
1. Four separate scripts with duplicate imports
2. Different metrics functions per scene
3. Hard-coded file paths
4. No way to run scenes together
5. Inconsistent code style

### Solutions Implemented:
1. **Unified imports** - Single import section, no duplication
2. **MetricsComputer class** - Centralized metric calculation
3. **Parameterized paths** - Scene classes accept image paths as arguments
4. **Pipeline orchestrator** - `ImageSegmentationPipeline` manages all scenes
5. **Class-based architecture** - Each scene is a standardized class with consistent methods

### What Was Preserved:
- ✅ All original algorithm logic (no rewriting)
- ✅ Variable names and structure (mostly unchanged)
- ✅ Metric calculations (identical formulas)
- ✅ Visualization approach (same matplotlib structure)
- ✅ Preprocessing pipelines (unchanged)

---

## How to Run the Program

### Basic Usage

```bash
# Terminal/Command Prompt
cd "C:\Users\ACER\OneDrive\Desktop\TAI FD"
python image_segmentation_unified.py all
```

### Python Script Usage

```python
from image_segmentation_unified import ImageSegmentationPipeline

# Create pipeline
pipeline = ImageSegmentationPipeline()

# Run specific scene
pipeline.run_scene(1)

# Or run all scenes
pipeline.run_all()
```

### Expected Console Output

```
==============================================================
SCÈNE 1 : Chat / Ciel / Sol / Arbres (K-Means k=4)
==============================================================
[Scene 1] Image: (480, 640, 3), GT: (480, 640, 3)
[Scene 1] Prétraitement terminé
[Scene 1] K-Means (k=4) convergé en 45 itérations
  Cluster 0 → sol
  Cluster 1 → ciel
  Cluster 2 → chat
  Cluster 3 → arbres

==================================================
  [SOL] - Scène 1
==================================================
  IoU            : 0.7234  ████████████████
  Dice           : 0.8401  ██████████████████
  Precision      : 0.8932  ███████████████████
  Recall         : 0.7812  ████████████████

...
```

---

## Metrics Interpretation

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth. 0.7+ is good, 0.9+ is excellent.
- **Dice (F1-Score)**: Harmonic mean of precision and recall. 0.8+ is very good.
- **Precision**: Of pixels predicted as target, how many are correct? 0.85+ is good.
- **Recall**: Of actual target pixels, how many were found? 0.8+ is good.

**Target ranges**: IoU > 0.7, Dice > 0.8, Precision > 0.85, Recall > 0.8

---

## Future Improvements

1. Add batch processing mode for multiple image sets
2. Create configuration file for scene parameters
3. Add logging instead of print statements
4. Implement performance profiling
5. Add data augmentation support
6. Create web interface for visualization
7. Export metrics to CSV/JSON

---

## Project Credits

- **Original Authors**: Aymen (Scene 1), Contributors (Scenes 2-4)
- **Module**: TAI & FD1 (Image Processing & Data Mining)
- **University**: [Institution Name]
- **Year**: 2025/2026

---

## License

This project is for educational purposes. All code and documentation are provided as-is.

---

## Questions & Support

For issues or questions:
1. Check the **CLAUDE.md** file for architecture details
2. Review the **Troubleshooting** section above
3. Check that all input images exist and are valid
4. Verify that all dependencies are installed

---

## Summary

✅ **4 scenes successfully merged into 1 unified program**
✅ **All original functionality preserved**
✅ **Consistent, scalable architecture**
✅ **Easy to run independently or together**
✅ **Comprehensive documentation**

**Ready for deployment!**
