# Week 3 Feedback — BreastMNIST (Corrected Submission)

Thanks for the correction update, it’s genuinely good that you caught issues in the Week 3 pipeline and re-submitted a more careful report. The new README is much stronger: you fixed several important methodological gaps and added the right evaluation metrics for an imbalanced medical dataset.

## What improved 

Cleaner, more correct preprocessing: You clearly documented the move to transforms.v2, proper float scaling, resizing to match ResNet input, and explicit grayscale → RGB channel handling.

Much better evaluation for imbalance: You added balanced accuracy, per-class recall (malignant recall), and PR-AUC, which fixes the “accuracy trap” on a 1:3 dataset.

More reliable conclusions: You now report mean ± std over 5 seeds, so augmentation comparisons are far more trustworthy.

Earlier pipeline/logging issues addressed: The updated metrics and reporting are more consistent and the analysis matches the numbers much better.

---

## High-priority fixes / clarifications

### 1) Numerical/text inconsistencies in the README

There are a few small places where the narrative doesn’t match the numbers:

* You wrote: “test accuracy does remain relatively high at 84% for `All_basic`, but `metrics_basic_all.json` shows **0.8974** (~90%).

* In AugB description, you wrote AUROC “decreases … (0.895 → 0.906)” — **0.906 is higher** than 0.895.


### 2) Head-only AUROC < 0.5 is still worth one explicit check

Even though your diagnosis “predicts only benign” is correct (malignant recall = 0), AUROC = **0.439** can also happen if:

* AUROC is computed using the **wrong class probability** (positive class flipped), or
* label encoding (malignant vs benign) is reversed in the AUROC function input.

Because your confusion-matrix-style reasoning matches collapse-to-majority behavior, this might be fine — but add **one sentence** in the README confirming you verified AUROC uses the probability for the **positive/malignant** class.

---

## Interpretation (mostly solid, with one nuance)

### 3) Your augmentation results are useful — phrase the conclusion carefully

From your 5-seed table:

* **All_basic** is best for clinically relevant objectives:

  * balanced acc **0.850 ± 0.012**
  * malignant recall **0.767 ± 0.026**
  * PR-AUC **0.818 ± 0.022**

* Augmentations (A/B/C) slightly improve calibration (ECE ↓), but they **reduce malignant recall** and balanced accuracy on average.

This is a great result and exactly what we want before SSL:

> Augmentations that help calibration can hurt minority-class sensitivity.

**Nuance:** ECE values are still relatively high overall (for example, All_basic ECE ~0.324). That suggests the model is not well-calibrated even in the best setting. Not “wrong,” but worth explicitly acknowledging, and later considering temperature scaling if calibration becomes a key evaluation axis.

---

## Suggested next steps (to make Weeks 4–6 smoother)

### 4) Choose the supervised baseline explicitly by objective

Right now you conclude `All_basic` is best, correct for **balanced accuracy / malignant recall / PR-AUC**.

A strong way to frame this for later SSL comparisons:

* **Baseline for sensitivity / detection:** `All_basic`
* **Baseline for calibration:** potentially `All_AugB` (lowest mean ECE in your table)

That will make it easier later to report whether SSL improves **discrimination**, **calibration**, or both.

### 5) Add 1–2 lines about why validation loss spikes are extremely large

Your val losses spike massively early (for example, 26, 18, 120). This can happen with tiny validation sets + unstable early logits, but it’s worth sanity-checking:

* evaluation uses `model.eval()` + `torch.no_grad()`
* loss reduction is averaged correctly
* labels are correct dtype/shape

You don’t need to fully debug it in the README, but adding “we verified eval mode/no_grad + correct label dtype” would increase confidence.

---


