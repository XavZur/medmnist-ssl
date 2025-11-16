The run completed and is reproducible. There seems no problem with  your implementation,  rather the approach. (I'm going to look over this one) However, AUROC ≈ 0.54 is near-random for binary classification, while Acc ≈ 0.71 suggests thresholding (or class imbalance) is masking poor discrimination. ECE ≈ 0.108 is relatively low, but calibration doesn’t matter much if the ranking is weak. Treat this baseline as a diagnostics case rather than a performance bar.

# What's likely going on
Underfitting/weak features: Head-only fine-tuning for 5 epochs on BreastMNIST often isn’t enough

Preprocessing mismatch: If 1-channel images weren’t properly converted to 3-channel, ResNet features degrade.

Metric wiring: AUROC uses the positive-class probability column; if label mapping or the column is wrong, AUROC tanks.


# Checks:
Best checkpoint actually used on test - Reload the best-val-AUROC weights before testing. Testing the final epoch by mistake often hurts

Confirm your transform includes Resize(64,64) → ToTensor() → 1→3 channel repeat: T.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x)

Maybe if you can find MedMNIST's official BreastMnist train/val/test splits, use MedMNIST’s official train/val/test splits


# Maybe next:
we can set FINETUNE='all', reduce LR to 1e-4 (maybe 5e-5), keep weight decay 1e-4, run 8–10 epochs.
