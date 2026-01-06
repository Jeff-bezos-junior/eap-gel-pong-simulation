# Model 2 — Current-based phenomenological Pong model (Primary model)

This directory contains **Model 2**, which is the **main model** analyzed in the thesis.

## Model concept
Model 2 directly follows the experimental Pong setup used in prior EAP-gel studies.
The paddle position is determined from **three region currents (A, B, C)** rather than
from explicit learning variables.

Key characteristics of this model are:

- Each region generates a current value
- The **stimulated region** produces a structured current response
- **Non-stimulated regions** produce low-amplitude noise
- Paddle position is calculated by fitting a **quadratic (parabolic) curve**
  to the three normalized current values
- The **vertex of the parabola** determines the target paddle position

This model allows direct investigation of how **current magnitude differences**
and their **time evolution** affect Pong performance.

## Script
- `Thesis_Model2.py`  
  Main simulation script for Model 2.
  Implements current generation, quadratic fitting, paddle control,
  and optional logging of current values.

## How to run
```bash
python Thesis_Model2.py
