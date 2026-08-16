# Clock Drawing CNN Analysis

A machine learning pipeline for automated analysis of clock drawing tests using convolutional neural networks. The pipeline preprocesses scanned clock drawings, builds a labeled dataset, and trains a CNN classifier to detect cognitive impairment indicators.

> **Key finding:** When transcript + time features are combined with image input, classification accuracy is highest. The best binary model achieves **Test AUC: 0.9237** (affected vs. not affected).

---

## Pipeline Overview

```
Raw .tif scans
      │
      ▼
preprocessing/clock_drawings/image_preprocess.ipynb   ← Binarize, clean artifacts, comparison output
      │
      ▼
preprocessing/clock_drawings/clock_processor.ipynb    ← Clean + island-crop to isolate clock drawing
      │
      ▼
dataset/clock_drawings/create_dataset.py              ← 80/10/10 train/test/valid split
      │
      ▼
model/clock_drawings/cnn_analysis_0to1.ipynb          ← Binary classification (affected vs. not)
model/clock_drawings/cnn_analysis_0to5.ipynb          ← 6-class classification (scores 0–5)
```

---

## Files

| File | Description |
|------|-------------|
| `preprocessing/clock_drawings/image_preprocess.ipynb` | Initial preprocessing: converts to B&W, removes dirt/rectangles/edge artifacts, saves side-by-side comparison TIFs |
| `preprocessing/clock_drawings/clock_processor.ipynb` | Full preprocessing: same cleaning as above + island-crop to isolate the clock, saves both comparison and cropped output |
| `dataset/clock_drawings/create_dataset.py` | Splits cropped TIFs into train/test/valid folders under `dataset/clock_drawings/processed` |
| `model/clock_drawings/cnn_analysis_0to1.ipynb` | Binary CNN: classifies as *affected* (scores 0–2) vs. *not affected* (scores 3–5) |
| `model/clock_drawings/cnn_analysis_0to5.ipynb` | Multiclass CNN: classifies into 6 score categories (0–5) |

---

## Preprocessing Details

### Cleaning Steps (both notebooks)
1. **Binarization** — Convert to grayscale, threshold at 128 → pure black/white
2. **Dirt removal** — Blobs smaller than `DIRT_THRESHOLD` pixels are erased
3. **Rectangle removal** — Dense rectangular regions (≥ `RECTANGLE_DENSITY`) are erased (e.g., censor bars, stamps)
4. **Edge artifact removal** — Any blob within `EDGE_THRESHOLD` pixels of the image border is erased

### Island Crop (`clock_processor.ipynb` only)
Finds the largest contiguous region of black pixels (tolerating gaps up to `CHASM_THRESHOLD`), crops a square bounding box around it with `BORDER_SIZE` padding, and resizes to `OUTPUT_SIZE × OUTPUT_SIZE`.

### Key Parameters

| Parameter | `image_preprocess` | `clock_processor` |
|-----------|-------------------|-------------------|
| `DIRT_THRESHOLD` | 50 | 50 |
| `RECTANGLE_DENSITY` | 0.95 | 0.95 |
| `EDGE_THRESHOLD` | 50 | 50 |
| `CHASM_THRESHOLD` | 30 | 30 |
| `BORDER_SIZE` | 10 | 20 |
| `OUTPUT_SIZE` | 320 | 640 |

---

## CNN Architecture

Both models use the same architecture:

```
Input: (N, 1, H, W) grayscale
  Conv2d(1→32, 3×3) + ReLU + MaxPool2d(2)
  Conv2d(32→64, 3×3) + ReLU + MaxPool2d(2)
  Conv2d(64→64, 3×3) + ReLU + AdaptiveAvgPool2d(4)
  Flatten → Linear(1024→128) + ReLU + Dropout(0.5)
  Linear(128 → num_classes)
```

- **Optimizer:** Adam (lr=1e-3)
- **Loss:** CrossEntropyLoss with inverse-frequency class weights
- **Epochs:** 20
- **Batch size:** 64
- **Downsampling:** 2× (images halved before training)

---

## Dataset

| Split | Samples |
|-------|---------|
| Train | 20,602 |
| Valid | 2,168 |
| Test | 1,412 |

Labels are read from `_classes.csv` in each split folder. Columns 2–7 correspond to scores 0–5; the argmax determines the raw label.

- **Binary model:** scores 0–2 → *affected* (1), scores 3–5 → *not affected* (0)
- **Multiclass model:** raw score used directly (0–5)

---

## Results

### Binary Model (`cnn_analysis_0to1.ipynb`)

| Metric | Value |
|--------|-------|
| Test AUC | **0.9004** |

**Confusion matrix (%):**

|  | Predicted: Not Affected | Predicted: Affected |
|--|------------------------|---------------------|
| **Actual: Not Affected** | 81.6% (TN) | 18.4% (FP) |
| **Actual: Affected** | 16.4% (FN) | 83.6% (TP) |

---

### Multiclass Model (`cnn_analysis_0to5.ipynb`)

| Metric | Value |
|--------|-------|
| Multiclass AUC (OVR macro) | **0.8625** |
| Binary AUC (derived) | **0.9237** |

**Confusion matrix — binarized post-hoc (%):**

|  | Predicted: Not Affected | Predicted: Affected |
|--|------------------------|---------------------|
| **Actual: Not Affected** | 87.2% (TN) | 12.8% (FP) |
| **Actual: Affected** | 27.1% (FN) | 72.9% (TP) |

> The multiclass model's derived binary AUC (0.9237) exceeds the dedicated binary model (0.9004), suggesting richer score-level supervision helps the model learn better representations.

---

## Notes

- Adding **transcript + time** features alongside the image was found to improve accuracy further (tested separately).
- Preprocessing skips already-processed files, making reruns safe and incremental.
- The `image_preprocess.ipynb` notebook saves only comparison TIFs (no crop); use `clock_processor.ipynb` for the full pipeline including cropped outputs.
