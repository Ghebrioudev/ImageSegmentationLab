# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a university image processing and data mining project (TAI & FD1) that performs image segmentation across 4 different scenes using various techniques. The project combines work from multiple contributors, each implementing one scene independently.

**Current Status:** 4 separate Python scripts (seg.py, projet_tai_fd_scene2.py, scene3_routes.py, mapartie.py) that need to be merged into one unified program.

## Project Structure

```
TAI FD/
├── seg.py                      # Scene 1: Chat/Ciel/Sol/Arbres (K-Means, k=4)
├── projet_tai_fd_scene2.py     # Scene 2: Retinal Bright Disk (Otsu + morphology)
├── scene3_routes.py            # Scene 3: Aerial Roads (K-Means, k=3)
├── mapartie.py                 # Scene 4: Person Extraction (Multi-channel K-Means, k=3)
├── Scene_1.png, GT1.png        # Input images for Scene 1
├── Scene_2.png, GT2.png        # Input images for Scene 2
├── Scene_3.png, GT3.png        # Input images for Scene 3
└── Scene_4_RGB_1.png, Scene_4_D_2.png, GT4.png  # Input images for Scene 4
```

## Architecture & Key Concepts

### Image Segmentation Pipeline (Common Pattern)

Each scene follows this general pipeline:
1. **Image Loading**: Load RGB/BGR images and ground truth masks
2. **Preprocessing**: Noise reduction (Gaussian blur, CLAHE, median filter, etc.)
3. **Segmentation**: Apply clustering or thresholding (K-Means, Otsu, top-hat, etc.)
4. **Post-Processing**: Morphological operations to refine masks
5. **Evaluation**: Compute metrics (IoU, Dice, Precision, Recall, Accuracy)
6. **Visualization**: Display results side-by-side with ground truth

### Scene-Specific Details

| Scene | Task | Method | Clusters | Key Preprocessing |
|-------|------|--------|----------|-------------------|
| 1 | Chat/Ciel/Sol/Arbres | K-Means | k=4 | Gaussian + RGB→LAB + MinMax normalization |
| 2 | Retinal Bright Disk | Otsu threshold | Binary | Green channel + CLAHE + Median + Top-hat |
| 3 | Aerial Roads | K-Means | k=3 | Gaussian + RGB→LAB |
| 4 | Person Extraction | K-Means (multi-channel) | k=3 | Gaussian + [R,G,B,D] feature matrix |

### Metrics Computation

All scenes use the same metrics but Scene 4 uses sklearn functions while others compute manually:
- **IoU** (Intersection over Union): TP / (TP + FP + FN)
- **Dice** (F1-Score): 2*TP / (2*TP + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **Accuracy** (Scene 4 only): (TP + TN) / (TP + TN + FP + FN)

### Key Libraries Used

- **opencv-python (cv2)**: Image I/O, preprocessing (blur, morphology, threshold), color space conversion
- **numpy**: Array operations, image manipulation
- **matplotlib**: Visualization and result display
- **scikit-learn (sklearn)**: K-Means clustering, metrics computation
- **scikit-image**: Advanced image processing (Scene 2: CLAHE)

## Important Coding Notes

### Image Format Conventions

- OpenCV reads images in **BGR format** by default → convert to RGB for matplotlib display
- Ground truth masks are typically **binary** (0 or 255)
- Normalized feature matrices should be in **[0, 1]** range for K-Means

### Critical Implementation Details

1. **Color Space Conversions**:
   - Scene 1 & 3 use LAB color space for better color separation than RGB
   - Scene 2 uses green channel extraction for retinal disk detection
   - Scene 4 concatenates RGB + depth into 4-channel features

2. **K-Means Usage**:
   - Always normalize input pixels (MinMaxScaler or division by 255)
   - Use `k-means++` initialization for stable convergence
   - Different scenes use different k values (Scene 1: k=4, Scene 3&4: k=3)

3. **Morphological Operations**:
   - Scene 2 uses elliptical kernel (35x35) for top-hat
   - Scene 4 uses elliptical kernel (7x7) for open/close operations
   - Scene 3 uses small (3x3) kernels for basic cleanup

4. **Ground Truth Matching**:
   - Scene 1: Matches clusters by dominant GT color in each region
   - Scene 2, 3, 4: Binary extraction by color ranges or thresholding

5. **File Path Handling**:
   - Current scripts use hard-coded paths (/content/ for Colab, local paths)
   - Need to be parameterized when merging

### Merging Strategy

When consolidating these scripts:
1. Create a unified metrics module with shared functions
2. Use parameterized functions for each scene (not global script logic)
3. Create a main() orchestrator that runs scenes independently or together
4. Handle file paths via command-line arguments or configuration
5. Preserve original algorithm logic without heavy refactoring
6. Rename duplicate functions (e.g., compute_metrics_scene1, compute_metrics_scene2)

### Common Pitfalls to Avoid

- Don't convert BGR to RGB multiple times in pipeline
- Don't forget to reshape 1D arrays back to 2D after K-Means
- Ensure GT mask dimensions match predicted mask before metric computation
- Be careful with feature matrix construction (order matters: [R,G,B,D], not [B,G,R,D])

## Dependencies

Install required packages:
```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image
```

## Development Workflow

When making changes:
1. **Test scene independence**: Each scene should work standalone
2. **Preserve algorithm logic**: Minimal changes to core segmentation
3. **Validate metrics**: Compare output metrics with original scripts
4. **Check visualizations**: Ensure plots display correctly
5. **Document file paths**: Make clear which images are expected for each scene

## Testing & Validation

After modifications:
- Run each scene individually to verify it still works
- Compare metrics with original scripts (should be identical or very close)
- Verify visualizations display all intermediate steps
- Check that ground truth masks are properly loaded and resized
- Ensure no hard-coded paths prevent execution

## Notes for Future Work

- Consider adding command-line interface for scene selection
- Could extract scene-specific preprocessing into separate modules
- Metrics computation could be more vectorized for performance
- Consider adding error handling for missing image files
- Results currently saved to local/named directories; could centralize output
