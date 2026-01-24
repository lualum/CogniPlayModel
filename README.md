# CogniPlay: Multi-Modal Cognitive Decline Detection

CogniPlay is a comprehensive platform for detecting cognitive decline through multiple data modalities: speech analysis, clock drawing tests, and conversational language patterns. The system combines deep learning models to provide early detection of dementia and cognitive impairment.

## Overview

This repository contains three independent detection models:

1. **Speech Analysis Model** - Detects cognitive decline from audio recordings using Wav2Vec2
2. **Clock Drawing Test Model** - Analyzes hand-drawn clock images using CNN
3. **Conversational Language Model** - Evaluates speech transcripts using hierarchical LSTM

## Models

### 1. Speech Analysis (Wav2Vec2)

Fine-tuned Wav2Vec2-XLS-R model for dementia detection from speech audio.

**Performance:**

- Accuracy: ~85%
- AUC-ROC: ~0.81

**Training:**

```python
# See train_model.ipynb for full implementation
from transformers import Wav2Vec2ForSequenceClassification

# Model trained on 32-second audio clips
# Sampling rate: 16kHz
# Classes: [nodementia, dementia]
```

**Dataset Requirements:**

- NHATS (National Health and Aging Trends Study) data
- Audio files in WAV format
- 16kHz sampling rate
- ~32 seconds per clip

### 2. Clock Drawing Test (CNN)

Convolutional neural network for scoring clock drawing tests (0-5 scale).

**Performance:**

- Best validation accuracy: ~82%
- Test accuracy: ~68-74%

**Training:**

```bash
# Preprocess images
python preprocess.py --input-folder drawings \
                     --chasm-threshold 30 \
                     --border-size 20 \
                     --output-size 640

# Split data
python split.py

# Train model (see cnn_analysis.ipynb)
```

**Dataset Structure:**

```
split/
├── train/
│   └── _classes.csv  # Labels: score 0-5
├── valid/
└── test/
```

### 3. Conversational Language (Hierarchical LSTM)

Hierarchical LSTM model analyzing conversational patterns from the Pitt Cookie Theft corpus.

**Training:**

```python
# See analyze_cha.ipynb for full implementation
from train_model import AlzheimerDetectionPipeline

pipeline = AlzheimerDetectionPipeline(max_utterances=20)
results = pipeline.cross_validate(data, labels, n_splits=5)
```

**Dataset Requirements:**

- Pitt Cookie Theft corpus (.cha files)
- CHAT format transcriptions
- Separate folders for dementia/control groups

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cogniplay.git
cd cogniplay

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy
pip install gensim pillow scipy
pip install librosa soundfile
```

## Quick Start

### Speech Analysis

```python
from transformers import pipeline

# Load model
classifier = pipeline(
    "audio-classification",
    model="cogniplayapp/wav2vec2-large-xls-r-300m-dm32"
)

# Predict
result = classifier("path/to/audio.wav")
print(result)
# [{'label': 'LABEL_1', 'score': 0.85}]  # dementia detected
```

### Clock Drawing Test

```python
import torch
from PIL import Image
import numpy as np

# Load model
model = torch.load('best_clock_model.pth')
model.eval()

# Preprocess image
img = Image.open('clock_drawing.tif').convert('L')
img_array = np.array(img.resize((320, 320)))
img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

# Predict score (0-5)
with torch.no_grad():
    score = model(img_tensor).argmax(1).item()
print(f"Clock Drawing Score: {score}/5")
```

### Conversational Analysis

```python
from train_model import AlzheimerDetectionPipeline

# Load and preprocess .cha files
pipeline = AlzheimerDetectionPipeline()
pipeline.load_word2vec()

# Predict
predictions, probabilities = pipeline.predict(conversation_data)
```

## Data Preprocessing

### Audio Files

```bash
# Convert to 16kHz mono WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Clock Drawings

```bash
# Clean and crop drawings
python preprocess.py --input-folder drawings \
                     --dirt-threshold 50 \
                     --rectangle-density 0.95
```

### CHAT Transcriptions

```python
# Extract participant utterances
from train_model import extract_par_utterances_with_duration

data = extract_par_utterances_with_duration("transcript.cha")
```

## Dataset Sources

1. **NHATS**: [National Health and Aging Trends Study](https://nhats.org/)
2. **Pitt Cookie Theft**: [DementiaBank](https://dementia.talkbank.org/)
3. **Clock Drawings**: Custom annotated dataset (scores 0-5)
