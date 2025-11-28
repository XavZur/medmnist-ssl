# Week 1

Dataset: pneumoniamnist  
Model: smallcnn  
Config: epochs=5, batch_size=128, seed=42

---

## Issue & Fix

Label Shape Error 
   - Error: "RuntimeError: 0D or 1D target tensor expected, multi-target not supported."
   - Cause: MedMNIST provides labels shaped [B, 1], but CrossEntropyLoss requires [B].
   - Fix: Added y = y.squeeze(1).long() in train, val, and test loops.

## Results
- **Final Test Accuracy:** 0.6218
- **Final AUROC:** 0.6494
- **Calibration (ECE):** 0.11093095861948454


