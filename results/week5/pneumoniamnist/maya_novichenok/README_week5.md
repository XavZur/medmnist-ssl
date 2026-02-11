Week 5 — Linear Probes (PneumoniaMNIST)

Encoder checkpoint: `results/week4/pneumoniamnist/maya_novichenok/ssl_encoder.pt`  
Backbone: ResNet-18 encoder (frozen)  
Goal: Measure label efficiency by training a linear head on top of a frozen SSL encoder.

Setup
- Input resolution: 224×224
- Channel handling: grayscale images converted to 3-channel for ResNet
  - `Grayscale(num_output_channels=3)`
- Normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
- Evaluation split: test split
- Metrics: Accuracy, AUROC (binary)

Experiments (core)
Linear probe (frozen encoder)
Trained a linear classifier on top of a frozen encoder using stratified label subsets.

- **5% labels:** trained on a stratified 5% subset of the train split  
  - Test Acc: 0.625
  - Test AUROC: 0.756

- **10% labels:** trained on a stratified 10% subset of the train split  
  - Test Acc: 0.763
  - Test AUROC: 0.862

Artifacts:
- `week5_probe_metrics.json` (saved metrics for 5% + 10%)

## Comparison vs supervised-only baseline
Compared probe results to the supervised-only baseline selected in Week 3 (best config from Week 3 for this dataset).

- Week 3 supervised baseline (reference):  
  - Test Acc: 0.83
  - Test AUROC: 0.961

## Takeaways
- With the encoder frozen, performance improves moving from 5% → 10% labels (expected), though performance is much better when the model is trained on the entire dataset.
