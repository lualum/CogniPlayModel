# CogniPlay: Multi-Modal Cognitive Behavior Classification

CogniPlay is a comprehensive platform for classifying cognitive behavior through multiple data modalities: speech analysis, clock drawing tests, and conversational language patterns. The system combines deep learning models to provide early classification of dementia and cognitive impairment.

## Overview

This repository contains maintained training and analysis entry points for:

1. **Speech Analysis Models** - Classify cognitive behavior from transcripts and timing features
2. **Cognitive Games Model** - Classify impairment from HRS/HCAP cognitive task scores
3. **Multimodal Fusion Analysis** - Combines modality-level predictions for cognitive classification experiments


## Repository Structure

The project is organized by pipeline stage, with modality-specific files inside each stage:

| Stage | Purpose | Key paths |
| ----- | ------- | --------- |
| `preprocessing/` | Feature extraction and raw-input cleanup | `preprocessing/speech/analyze_cha_stats.py` |
| `dataset/` | Source data | `dataset/speech/Pitt/`, `dataset/clocks/ClockData/`, `dataset/games/HC22/` |
| `model/` | Training, modeling, and fusion code | `model/speech/`, `model/clocks/`, `model/games/`, `model/fusion/` |
| `output_performance/` | Generated predictions, metrics, plots, and performance artifacts | `output_performance/speech/cha_stats/`, `output_performance/speech/pitt_transformer/`, `output_performance/clocks/nhats_cnn/`, `output_performance/games/hcap/`, `output_performance/fusion/` |

### Common Entry Points

| Task | Command or notebook |
| ---- | ------------------- |
| Build speech segmentation features and model-input checks | `python preprocessing/speech/analyze_cha_stats.py` |
| Resize NHATS clock images to 256x256 | `python preprocessing/clocks/resize_clock_images.py` |
| Train/run the speech baseline model | `python model/speech/train_speech_model.py` |
| Train Pitt speech transformer classifier | `python model/speech/train_pitt_transformer.py` |
| Train NHATS clock drawing CNN | `python model/clocks/train_clock_cnn.py` |
| Train HCAP cognitive games classifier | `python model/games/train_hcap_games_model.py` |
| Generate ROC/AUC fusion figures from prediction files | `python model/fusion/multimodal_fusion_bayes.py` |
| Generate PyTorch architecture figures | `python model/generate_model_architecture_figures.py` |


## Results

### Multimodal Fusion Performance

The fusion script reads the current prediction CSVs and computes held-out test ROC/AUC curves. Because the speech, clock drawing, and games files contain no overlapping subject IDs, individual-modality curves are direct test-set measurements, while combined-modality curves are labeled score-distribution fusion estimates rather than true participant-level fusion. True multimodal fusion requires prediction files for the same participants across modalities.

| Modality Configuration                       | AUC        |
| -------------------------------------------- | ---------- |
| Speech only                                  | 0.8871     |
| Clock Drawing only                           | 0.6809     |
| Games only                                   | 0.9146     |
| Speech + Clock Drawing + Games fusion estimate | **0.9933** |

Generated outputs are written under `output_performance/fusion/`, including `auc_summary.csv`, `auc_curves.csv`, `auc_metadata.json`, no-legend PNG/JPG/SVG figures, and separate `figure_legends.docx`, `figure_legends.txt`, and `figure_legends.md` files for journal submission.

### Fusion Weights

The current score-distribution fusion estimate selects the following held-out probability weights:

| Modality      | Fusion Weight |
| ------------- | ------------- |
| Games         | 50.0%         |
| Clock Drawing | 35.0%         |
| Speech        | 15.0%         |

These weights should be interpreted as a model-score experiment, not a clinical multimodal estimate, until shared-subject prediction files are available.

---

## Models

### 1. Speech Analysis (LSTM/Transformer)


An LSTM/Transformer model trained on timestamped word sequences from speech transcripts to predict MMSE scores. Speech features were evaluated using transcription text alone, timing features alone, and a combination of both. The combined input produced the highest accuracy, as pause patterns and speaking-rate signals capture cognitive changes not reflected in text alone. See `model/speech/train_speech_model.py` and `model/speech/train_pitt_transformer.py` for implementation.


**Dataset:** [DementiaBank ADReSSo](https://dementia.talkbank.org/) — speech transcripts with timestamped word sequences.

### 2. Clock Drawing Test (CNN)


A CNN trained to predict a 0–5 clock drawing score from NHATS Round 14B clock images. Run `python preprocessing/clocks/resize_clock_images.py` to create 256x256 image copies under `dataset/clocks/ClockData_256/`; the CNN uses that smaller directory by default when present. The maintained training entry point is `model/clocks/train_clock_cnn.py`, which joins image filename subject ids to `spid` in `dataset/clocks/ClockData/NHATS_Round_14B_SP_File.sas7bdat` and uses `cg14dclkdlnn` as the default ground-signal column. It writes score predictions, binary impairment probabilities for fusion, metrics, and a PyTorch checkpoint under `output_performance/clocks/nhats_cnn/`.


**Dataset:** [NHATS](https://nhats.org/) Round 14 — clock drawing images with associated cognitive scores.

### 3. Cognitive Games (HRS/HCAP Classifier)


A classifier trained on HRS/HCAP cognitive test scores to predict MMSE, capturing memory and executive function. Labels were assigned based on MMSE threshold: scores <=23 (impaired) and scores >=24 (normal). The old exploratory notebooks have been removed from this repository cleanup.
The maintained training entry point is `model/games/train_hcap_games_model.py`, which writes Pitt-transformer-style predictions and metrics plus a fusion-compatible `True,Prob` CSV.


**Dataset:** [HRS/HCAP](https://hcap.isr.umich.edu/) — cognitive test scores used to derive MMSE predictions.

## Model Architecture Figures

Run `python model/generate_model_architecture_figures.py` to generate compact, shape-accurate architecture schematics for the maintained PyTorch models:

- `output_performance/model_architecture/clock_cnn_architecture.png`
- `output_performance/model_architecture/speech_transformer_architecture.png`

SVG versions are generated alongside the PNGs, and the former `*_visualtorch.png` paths are retained as compatibility copies. The script uses saved checkpoint config/vocab when present and otherwise falls back to default model settings. Install the Python dependencies from `requirements.txt` before rendering.

## Dataset Sources

1. **DementiaBank ADReSSo**: [dementia.talkbank.org](https://dementia.talkbank.org/)
2. **NHATS Round 14**: [National Health and Aging Trends Study](https://nhats.org/)
3. **HRS/HCAP**: [Health and Retirement Study / Harmonized Cognitive Assessment Protocol](https://hcap.isr.umich.edu/)
