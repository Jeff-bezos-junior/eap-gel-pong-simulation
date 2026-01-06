# Model 1 — Neural-inspired learning (sigmoid / threshold adaptation)

This directory contains the code for **Model 1** from the thesis.

## Concept (what this model represents)
Model 1 is a **neural-inspired abstraction** of learning-like improvement.  
Each region (A/B/C) accumulates stimulation history, and the system becomes more responsive over time by:

- decreasing an effective response threshold, and/or  
- increasing response probability

using a **sigmoid-shaped learning curve**.

This model is intended to demonstrate **learning-like performance changes** in Pong in a simplified way. Note that it is **not** a direct reproduction of measured EAP-gel current waveforms.

## Files
- `Thesis_Model1.py`  
  Main simulation script for Model 1 (Pong + region learning mechanism).
  Runs an interactive Turtle-based Pong simulation and updates region-dependent internal learning variables during play.

## How to run
```bash
python Thesis_Model1.py
