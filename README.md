# CogniPlay: Multi-Modal Cognitive Behavior Classification

CogniPlay is a comprehensive platform for classifying cognitive behavior through multiple data modalities: speech analysis, clock drawing tests, and conversational language patterns. The system combines deep learning models to provide early classification of dementia and cognitive impairment.

## Overview

This repository contains three independent classification models:

1. **Speech Analysis Model** - Classifies cognitive behavior from speech transcripts and timing features using an LSTM/Transformer
2. **Clock Drawing Test Model** - Analyzes hand-drawn clock images using a CNN
3. **Cognitive Games Model** - Predicts MMSE scores from HRS/HCAP cognitive test scores

## Results

### Multimodal Fusion Performance

Combining modalities improves classification performance compared with individual models. The table below summarizes AUC scores across all configurations:

| Modality Configuration                       | AUC        |
| -------------------------------------------- | ---------- |
| Speech only                                  | 0.6021     |
| Clock Drawing only                           | 0.6483     |
| Games only                                   | 0.9577     |
| Speech + Clock Drawing + Games (full fusion) | **0.9628** |

Among single modalities, games achieved the strongest performance (AUC 0.9577), while clock drawing and speech alone achieved lower AUCs of 0.6483 and 0.6021 respectively. The full multimodal fusion of all three modalities achieved the highest AUC of 0.9628.

### Modality Contribution Analysis

Although games contributed the majority of predictive power, the addition of speech and clock drawing provided complementary information that improved overall accuracy:

| Modality      | Predictive Contribution |
| ------------- | ----------------------- |
| Games         | 74.9%                   |
| Clock Drawing | 15.6%                   |
| Speech        | 9.6%                    |

These findings support the hypothesis that multimodal fusion can enhance classification accuracy by integrating diverse behavioral signals such as speech patterns, drawing performance, and cognitive task outcomes. The incremental gains from including speech and clock drawing demonstrate that each modality captures a distinct and complementary dimension of cognitive function.

Future work should investigate performance across larger datasets and explore whether additional signal types could further improve classification accuracy.

---

## Models

### 1. Speech Analysis (LSTM/Transformer)

An LSTM/Transformer model trained on timestamped word sequences from speech transcripts to predict MMSE scores. Speech features were evaluated using transcription text alone, timing features alone, and a combination of both. The combined input produced the highest accuracy, as pause patterns and speaking-rate signals capture cognitive changes not reflected in text alone. Synonym replacement was applied for data augmentation to improve generalization. See `train_model.ipynb` for full implementation.

**Performance:**

- Accuracy: ~85%
- AUC-ROC: ~0.81

**Dataset:** [DementiaBank ADReSSo](https://dementia.talkbank.org/) — speech transcripts with timestamped word sequences.

### 2. Clock Drawing Test (CNN)

A CNN trained to predict a cognitive score from 0–5 based on clock drawing images. A continuous score prediction approach proved more effective than binary classification. Images were preprocessed to remove noise and artifacts before training. See `cnn_analysis.ipynb` for full implementation.

**Performance:**

- Best validation accuracy: ~82%
- Test accuracy: ~68-74%

**Dataset:** [NHATS](https://nhats.org/) Round 14 — clock drawing images with associated cognitive scores.

### 3. Cognitive Games (HRS/HCAP Classifier)

A classifier trained on HRS/HCAP cognitive test scores to predict MMSE, capturing memory and executive function. Labels were assigned based on MMSE threshold: scores ≤23 (impaired) and scores ≥24 (normal). See `analyze_cha.ipynb` for full implementation.

**Dataset:** [HRS/HCAP](https://hcap.isr.umich.edu/) — cognitive test scores used to derive MMSE predictions.

## Dataset Sources

1. **DementiaBank ADReSSo**: [dementia.talkbank.org](https://dementia.talkbank.org/)
2. **NHATS Round 14**: [National Health and Aging Trends Study](https://nhats.org/)
3. **HRS/HCAP**: [Health and Retirement Study / Harmonized Cognitive Assessment Protocol](https://hcap.isr.umich.edu/)
