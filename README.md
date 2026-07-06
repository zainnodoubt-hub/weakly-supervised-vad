# Weakly Supervised Video Anomaly Detection

This repository contains a five-notebook pipeline for weakly supervised video anomaly detection using UCF-Crime and XD-Violence.

## Notebooks

1. `01data_pipeline.ipynb` — prepares metadata, dataset splits, and configuration.
2. `02features_dataloader.ipynb` — builds fixed-length feature bags, manifests, and PyTorch data loaders.
3. `03baseline_mil_loss.ipynb` — defines the MIL models, pooling methods, losses, and evaluation helpers.
4. `04attention_training.ipynb` — trains and compares baseline, attention, and CLIP-semantic MIL models.
5. `05evaluation.ipynb` — evaluates saved models and produces final visualisations and report figures.

Large datasets, extracted features, model checkpoints, experiment runs, environments, and generated exports are intentionally excluded from version control.
