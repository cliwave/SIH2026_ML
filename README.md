# SIH2026_ML

## Smart Water Quality Pattern Detection

This repository contains the Machine Learning module developed for **SIH 2026** under the problem statement:

**Smart Water Purification and Quality Monitoring System for Rural and Mining-Affected Areas**

The ML module focuses on recognizing unusual patterns in water-quality measurements using four parameters:

- pH
- Turbidity
- TDS (Total Dissolved Solids)
- Temperature

## How the ML Module Works

The system receives the four water-quality parameters as input and analyzes them together to identify whether the observed pattern is:

- **NORMAL**
- **UNUSUAL**

The current prototype uses **Isolation Forest**, an unsupervised machine-learning algorithm for anomaly detection.

The four parameters are processed together rather than using four separate ML models.

### Workflow

```text
Water Quality Readings
        ↓
pH + Turbidity + TDS + Temperature
        ↓
Isolation Forest
        ↓
Pattern Detection
        ↓
NORMAL / UNUSUAL
        ↓
Flask API Response