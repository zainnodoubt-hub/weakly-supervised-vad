# Weakly Supervised Video Anomaly Detection

This repository contains the implementation and experimental work for an **EEEM068 Applied Machine Learning** group project on **Video Anomaly Detection (VAD)** using weakly supervised deep learning methods.

The project focuses on detecting abnormal events in surveillance videos using only **video-level labels**, instead of precise frame-level annotations. This is a practical setting because annotating the exact start and end of anomalies in long surveillance videos is expensive and time-consuming.

---

## Project Overview

Video Anomaly Detection aims to identify unusual or abnormal events in video sequences, such as accidents, fighting, explosions, robbery, vandalism, or other suspicious activities.

In this project, each video is treated as a **bag** of temporal segments, and each segment is treated as an **instance**. Since only video-level labels are available, the model learns to assign anomaly scores to segments through a **Multiple Instance Learning (MIL)** framework.

The main objectives of this project are to:

- Build a weakly supervised VAD pipeline.
- Use pre-extracted spatiotemporal video features.
- Implement MIL-based anomaly detection models.
- Improve the baseline using attention and temporal modelling.
- Evaluate models using anomaly detection metrics.
- Analyse anomaly score curves and model behaviour.
- Explore a CLIP-based semantic extension for extra analysis.

---

## Datasets

This project uses feature representations from two video anomaly detection datasets:

### UCF-Crime

UCF-Crime is a large-scale real-world surveillance video dataset containing normal and anomalous videos across multiple anomaly categories.

### XD-Violence

XD-Violence is a large-scale weakly supervised violence detection dataset containing videos from multiple scenarios. In this project, RGB and optical-flow feature bags are used where available.

> **Note:** Raw datasets, extracted feature files, model checkpoints, and large result files are not included in this repository due to size and licensing restrictions. They should be downloaded or generated separately and placed in the correct data folders.

---

## Methodology

The project follows a weakly supervised learning setup:

1. Videos are divided into fixed temporal segments.
2. Each video is represented as a bag of segment-level features.
3. A MIL model predicts anomaly scores for each segment.
4. Video-level labels supervise the model during training.
5. Model performance is evaluated using video-level and anomaly detection metrics.

The implemented approaches include:

- **Baseline MIL model**
- **Attention MIL model**
- **BiGRU-MIL model**
- **CLIP semantic extension**
- **Top-k MIL aggregation**
- **Anomaly score curve visualisation**
- **Feature-space visualisation using UMAP/t-SNE**

---

## Models

### Baseline MIL

A Sultani-style weakly supervised MIL baseline that learns to assign higher anomaly scores to abnormal video segments than normal segments.

### Attention MIL

An improved MIL model that uses attention to learn which temporal segments are more important for anomaly detection.

### BiGRU-MIL

A temporal sequence model that captures relationships between neighbouring video segments using bidirectional recurrent modelling.

### CLIP Semantic Extension

An additional semantic component that uses CLIP-style text-video semantic alignment to support anomaly understanding and improve interpretability.

---

## Evaluation Metrics

The main evaluation metrics used in this project are:

- **ROC-AUC**
- **Average Precision (AP)**
- **Precision-Recall analysis**
- **Anomaly score curves**
- **Inference-time comparison**
- **Qualitative analysis of true positives, false positives, and failure cases**

---

## Experimental Results

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Baseline MIL | 0.9347 | 0.9315 |
| Attention MIL | 0.9579 | 0.9492 |
| Attention + CLIP Semantic Loss | 0.9920 | 0.9924 |

The attention-based model improved over the baseline MIL approach, showing that learning temporal importance across segments is useful for weakly supervised anomaly detection. The CLIP semantic extension gave the strongest performance in the final experiments.

---

## Project Structure

```text
weakly-supervised-video-anomaly-detection/
│
├── data/                   # Dataset manifests or small metadata files
├── features/               # Pre-extracted feature bags (not uploaded if large)
├── notebooks/              # Jupyter notebooks for experiments and analysis
├── src/                    # Source code for models, training, and evaluation
│   ├── datasets/           # Dataset loading and preprocessing scripts
│   ├── models/             # MIL, Attention MIL, BiGRU-MIL, CLIP extensions
│   ├── training/           # Training loops and loss functions
│   ├── evaluation/         # Metrics and evaluation utilities
│   └── visualisation/      # Score curves, UMAP/t-SNE, plots
│
├── results/                # Tables, plots, and evaluation outputs
├── reports/                # Final report and related documentation
├── logs/                   # Training logs and experiment records
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Files and folders excluded from GitHub
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/weakly-supervised-video-anomaly-detection.git
cd weakly-supervised-video-anomaly-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Prepare the data

Place dataset manifests and extracted feature files in the expected folders:

```text
data/
features/
```

Large files should not be committed to GitHub.

### 2. Train a model

Example command:

```bash
python src/training/train.py --model attention_mil --dataset ucf_crime
```

### 3. Evaluate a model

Example command:

```bash
python src/evaluation/evaluate.py --model attention_mil --dataset ucf_crime
```

### 4. Generate visualisations

Example command:

```bash
python src/visualisation/plot_scores.py
```

---

## Key Features

- Weakly supervised video anomaly detection
- Multiple Instance Learning pipeline
- Attention-based temporal modelling
- BiGRU temporal modelling
- CLIP-based semantic extension
- UCF-Crime and XD-Violence support
- ROC-AUC and AP evaluation
- Anomaly score visualisation
- UMAP/t-SNE feature analysis
- Training logs and experiment tracking

---

## Limitations

- The model is trained using weak video-level labels, so exact temporal localisation is approximate.
- Frame-level ground truth is limited or unavailable for some experiments.
- Performance depends heavily on the quality of pre-extracted video features.
- Some false positives can occur when normal scenes visually resemble abnormal events.
- Large datasets and feature files are not included in the repository.

---

## Future Work

Possible improvements include:

- Fine-tuning video feature extractors end-to-end.
- Adding stronger temporal transformer-based models.
- Improving anomaly localisation with better temporal smoothing.
- Testing on additional VAD datasets.
- Extending the CLIP semantic branch for anomaly label prediction.
- Reducing inference time for real-time deployment.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- UMAP / t-SNE
- CLIP-based semantic features
- Jupyter Notebook

---

## Academic Context

This project was completed as part of the **EEEM068 Applied Machine Learning** module at the **University of Surrey**.

The project applies deep learning concepts including CNN-based video features, weak supervision, Multiple Instance Learning, attention mechanisms, temporal modelling, and multimodal semantic learning.

---

## Contributors

This was a group coursework project. Contributions include data preparation, model implementation, training, evaluation, visualisation, report writing, and presentation preparation.

---

## License

This repository is intended for academic coursework and learning purposes. Dataset usage is subject to the original dataset licences and access conditions.

---

## Acknowledgements

This project is based on research in weakly supervised video anomaly detection, including MIL-based methods for real-world surveillance anomaly detection and multimodal violence detection.
