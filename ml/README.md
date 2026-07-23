# Machine Learning

This directory contains the Machine Learning implementation for EVGuardian AI.

The ML pipeline will be used to analyze lithium-ion battery degradation and develop models for:

- State of Health (SOH) estimation
- Remaining Useful Life (RUL) prediction
- Battery degradation analysis
- Predictive maintenance

## Planned Workflow

1. Dataset loading and exploration
2. Data preprocessing
3. Feature selection and engineering
4. Model training
5. Model evaluation and comparison
6. SOH estimation
7. RUL prediction
8. Selection of the best-performing model
9. Deployment of the trained model for real-time inference

## Planned Structure

- `notebooks/` – Data exploration and experimental notebooks
- `preprocessing/` – Data cleaning and feature engineering
- `training/` – Model training scripts
- `models/` – Saved trained models

The final Machine Learning model will be integrated with the Raspberry Pi-based hardware system for real-time battery health analysis.
