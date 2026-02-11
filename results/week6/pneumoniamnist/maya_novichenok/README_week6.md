# Week 6 — SSL Fine-tuning + Calibration (PneumoniaMNIST)

Note: I unfortunately did not have the compute ability to run many epochs

Backbone: ResNet-18  
Goal: Compare supervised baseline vs SSL probe vs SSL fine-tuning, including calibration.

## Setup
- Input resolution: 224×224
- Channel handling: grayscale → 3-channel (`Grayscale(num_output_channels=3)`)
- Normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
- Metrics: Accuracy, AUROC, ECE (10 bins)
- Calibration plot: reliability diagram (saved as PNG)

## Models compared
### 1) Supervised-only baseline

- Test Acc: 0.867
- Test AUROC: 0.976
- Test ECE: null

### 2) SSL probe (frozen encoder)
Linear probe trained on top of a frozen SSL encoder (Week 5 result).

- Label fraction: **10%**
- Test Acc: 0.763
- Test AUROC: 0.862
- Test ECE: null

### 3) SSL fine-tune (unfrozen encoder, small LR)
Fine-tuned the SSL encoder end-to-end with a small encoder learning rate.

- Optimizer: AdamW
- Encoder LR: 1e-4
- Head LR: 1e-3
- Epochs: 2 (ideally would run more for more accurate results)

Results:
- Test Acc: 0.854
- Test AUROC: 0.953
- Test ECE: 0.143

Calibration:
- Reliability diagram saved to: `reliability_ssl_finetune.png`  
  (This satisfies the Week 6 requirement of ≥1 reliability plot for an SSL model.)

## Artifacts
- `week6_comparison.json` (baseline vs probe vs finetune metrics)
- `reliability_ssl_finetune.png` (reliability diagram)

## Takeaways
- Fine-tuning generally improves performance over a frozen probe at the same label fraction.
- Calibration (ECE + reliability) is evaluated explicitly for the SSL model to understand confidence quality, not just accuracy.

