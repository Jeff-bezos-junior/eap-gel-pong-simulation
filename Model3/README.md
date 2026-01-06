# Model 3 — Retention-based Pong model

This directory contains **Model 3**, which focuses on **response retention**
rather than learning.

## Model concept
Model 3 assumes that the system does **not learn over time**.
Instead, a fixed **retention factor** determines how strongly previous
responses persist and influence paddle motion.

Key characteristics:

- No learning variable or history-based update
- Paddle target position is derived directly from ball position
- Retention controls the **speed and smoothness** of paddle movement
- Higher retention → faster tracking
- Lower retention → slower tracking and more misses

This model isolates the effect of **material-like persistence**
independent of learning.

## Script
- `Thesis_Model3.py`  
  Main simulation script for Model 3.

## How to run
```bash
python Thesis_Model3.py
