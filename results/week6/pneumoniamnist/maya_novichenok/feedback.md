# What’s strong

Clear structure and correct Week 6 intent: you compare supervised baseline vs SSL probe vs SSL fine-tuning, and you explicitly include calibration (ECE + reliability diagram).

Training setup is documented well (224×224, grayscale→3ch, normalization, metrics + binning for ECE).

The trend you report is reasonable: fine-tuning improves performance vs frozen probe at the same label fraction (Acc 0.763 → 0.854, AUROC 0.862 → 0.953).

#  Main issues to fix (important for B1/B3 quality)
1) Calibration reporting is incomplete (ECE is null for baseline + probe)

Right now only SSL fine-tune has ECE, while:

supervised baseline ECE = null

SSL probe ECE = null

This prevents fair calibration comparison and weakens the Week 6 “calibration” requirement beyond “at least one plot”.

Actionable fix: compute and report ECE for all three models, using the same:

split (test vs val),

probability definition (positive-class probability),

binning (10 bins),

and reliability plotting function.

Even if you only plot SSL fine-tune, you should still report ECE for baseline and probe.

2) Baseline comparison may not be apples-to-apples (label fraction mismatch)

You compare:

SSL probe at 10% labels

SSL fine-tune at (implicitly) 10% labels? (not explicitly stated)

supervised baseline likely trained on 100% labels (unclear)

If the supervised baseline used full labels, it’s not directly comparable for “label efficiency”.

Actionable fix: explicitly state the supervised baseline training label fraction. If it’s 100%, add a supervised 10% label run (same backbone + resolution) so comparisons are fair.

3) Fine-tuning uses only 2 epochs → conclusions should be phrased cautiously

You correctly note compute limits, but 2 epochs can significantly under-train or lead to unstable results, especially for calibration.

Actionable fix options:

If you can’t run more epochs: add a sentence like

“Results are preliminary due to 2-epoch training; relative trends are more meaningful than absolute numbers.”

Or do a lightweight improvement: run 3 seeds with 2 epochs and report mean ± std (often easier than running 20 epochs once).

4) Clarify what probability is used for ECE and reliability diagram

ECE is sensitive to whether “confidence” means:

positive-class probability P(y=1), or

max softmax probability of predicted class

Actionable fix: add one line:

“ECE/reliability use the model’s predicted probability for the positive class (pneumonia).”

# Suggestions that would improve scientific quality (optional but high value)

Add one class-sensitive metric: recall/sensitivity for pneumonia, specificity, or confusion matrix.
PneumoniaMNIST is medical; accuracy alone can hide clinically important behavior.

Consider post-hoc calibration: temperature scaling on validation (easy and often improves ECE without hurting AUROC).
