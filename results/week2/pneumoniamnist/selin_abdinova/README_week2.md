## Dataset Recap
We use PneumoniaMNIST, a binary classification dataset of pediatric chest X-ray images with labels:
0: Normal
1: Pneumonia
The dataset is moderately class-imbalanced, with more pneumonia cases than normal cases in the test split, which makes accuracy alone insufficient as a metric and motivates the use of AUROC. Images are low-resolution grayscale scans that have been resized and normalized; many samples appear blurry and low-contrast, and anatomical structures (e.g., ribs, heart shadow) are not always clearly visible. Some images are visually ambiguous even for humans, suggesting inherent label difficulty.

## Baseline Comparison

| Model | Accuracy | AUROC | ECE |
|------|---------|-------|-----|
| SmallCNN | 0.6218 | 0.6494 | 0.1128 |
| ResNet18 (head fine-tuned) | 0.7628 | 0.8252 | 0.2389 |

## Comments:
ResNet18 substantially outperforms SmallCNN in both accuracy and AUROC, indicating better discriminative performance. This is expected, as pretrained convolutional features transfer well even to small medical datasets. However, SmallCNN achieves a lower ECE, suggesting that despite weaker classification performance, its confidence estimates are better calibrated.

## Calibration Snapshot
The ResNet18 reliability diagram yields an ECE of 0.239. The diagram shows that at higher confidence levels (≈0.8–1.0), the model is overconfident, with empirical accuracy falling below predicted confidence. This suggests that while ResNet18 makes correct predictions more often, its probability outputs are not well aligned with true correctness. In contrast, SmallCNN’s lower ECE indicates more conservative and better-aligned confidence estimates. This highlights a trade-off between performance and calibration in pretrained models.

## Error Analysis
Inspection of misclassified examples suggests:
- Several false positives correspond to normal X-rays with low contrast or shadowing, which may resemble pneumonia patterns.
- Some false negatives show very subtle pneumonia indicators, likely below the model’s resolution sensitivity.
- Many errors involve images that are visually ambiguous, indicating intrinsic dataset difficulty rather than purely model failure.
- Misclassifications often occur at moderate confidence, consistent with uncertainty in borderline cases.
