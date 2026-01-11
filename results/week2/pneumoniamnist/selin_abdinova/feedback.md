## Week 2 Feedback — PneumoniaMNIST (Student C)

Overall this is a good Week 2 submission: you computed all required metrics (Acc / AUROC / ECE), produced reliability diagrams, and did a careful qualitative error analysis. Below are some more detailed comments and suggestions.

---

### 1. Dataset Recap

- You clearly state the task (binary PneumoniaMNIST, 0 = Normal, 1 = Pneumonia).
- You highlight **moderate class imbalance** and correctly note that this makes “accuracy alone” insufficient and motivates AUROC.
- The description of image quality (low‐resolution, blurry, low contrast, ambiguous even for humans) is exactly the kind of EDA we want.

**Suggestions**

- If you have the exact **class counts per split** (train / val / test, normal vs pneumonia), consider adding a small table. It makes the imbalance more concrete and consistent across students.
- You can sharpen the metrics commentary slightly:

  > “Because pneumonia is more common than normal in the test split, a classifier that favors pneumonia can get a high accuracy while still failing to separate the classes well. AUROC is less sensitive to class imbalance and is therefore a better measure of discrimination here.”

---

### 2. Baseline Comparison

**Table recap**

| Model                    | Test Acc | Test AUROC | Test ECE |
|--------------------------|---------:|-----------:|---------:|
| SmallCNN                 | 0.6218   | 0.6494     | 0.1128   |
| ResNet18 (head fine-tune)| 0.7628   | 0.8252     | 0.2389   |

**Good points**

- You correctly conclude that **ResNet18 substantially outperforms SmallCNN in accuracy and AUROC**, which means much better discrimination.
- The explanation that **pretrained convolutional features transfer well** even on small medical datasets is on point.
- Noting that **SmallCNN has lower ECE** and thus better calibration, despite worse accuracy/AUROC, is an important observation.

**Suggestions**

- Make the AUROC gain explicit:

  > “AUROC improves from ≈0.65 (SmallCNN) to ≈0.83 (ResNet18), indicating that the deeper model separates pneumonia vs normal much better across thresholds.”

- You could briefly mention the **trade-off** in one sentence:

  > “ResNet18 trades better discrimination for poorer calibration, while SmallCNN is weaker but more conservative in its probabilities.”

---

### 3. Calibration Snapshot

**What you did well**

- You report **ECE ≈ 0.239** for ResNet18 and clearly describe the pattern in the reliability diagram: **over-confidence at high confidence (0.8–1.0)** where empirical accuracy falls below predicted confidence.
- You correctly interpret that SmallCNN’s lower ECE means its confidences are more conservative and better aligned.

**Linking more tightly to the plot**

From the reliability diagram:

- Most ResNet18 predictions lie in mid/high confidence bins (≈0.5–1.0).
- In bins around 0.7–0.9, accuracy is clearly below confidence → consistent, systematic over-confidence.
- The histogram bars help show that ECE is driven mainly by these high-confidence regions.

You can mention explicitly that:

> “Because most test samples fall into the high-confidence bins, even a modest gap there contributes a lot to the overall ECE ≈ 0.24.”

**Optional extensions**

- Specify the binning scheme: for example, “10 equal-width bins on [0, 1]”.
- If you have time later, you could add an **equal-frequency (quantile) reliability diagram** to check that the ECE conclusion is robust to binning.

---

### 4. Confusion Matrices & Error Analysis

You produced confusion matrices for both models and a gallery of “misclassified” examples.

- The **SmallCNN confusion matrix** looks reasonable (two numbers per row), and your error analysis matches the idea that many mistakes occur on low-contrast or subtle cases.
- Your qualitative bullets are good:

  - false positives: normal X-rays with low contrast or shadowing;
  - false negatives: very subtle pneumonia;
  - many images visually ambiguous → intrinsic difficulty, not just model failure.

**Two technical issues to fix**

1. In the ResNet18 confusion matrix image, it looks like **only one cell (233) is populated**, with the others empty. That suggests an indexing or plotting bug (like using only predictions from one class, or mixing up labels on the axes). It would be good to:

   - double-check how you compute `confusion_matrix(y_true, y_pred)`, and  
   - ensure you plot all four entries (TN, FP, FN, TP).

2. In the “misclassified gallery”, the titles all read **`T:[1] P:1`**, for example: true label 1, predicted label 1. That means you’re currently plotting **true positives**, not misclassifications. Likely the mask is something like `pred == true` instead of `pred != true`. Once you fix the mask, you’ll see genuine errors such as `T:0 P:1` (false positives) and/or `T:1 P:0` (false negatives), which will make your written error analysis even more concrete.

You can keep your current written bullets, but after fixing these two issues you’ll be able to directly point to specific examples in the gallery and say:

> “For example, this high-confidence false positive normal has bright central opacity and weak borders, visually resembling pneumonia.”

---

### 5. Summary & Next Steps

Your final summary already captures the right message:

- PneumoniaMNIST is imbalanced and low-resolution → intrinsically hard.
- ResNet18 is clearly better at **discrimination** (Acc + AUROC) but **worse calibrated** (higher ECE).
- SmallCNN is less powerful but more **honest** in its probabilities.

If you want one more sentence for Week 2:

> “In future weeks, it would be interesting to combine the strong representation of ResNet18 with explicit calibration methods (temperature scaling or label-efficiency experiments), to see if we can reduce ECE without sacrificing AUROC.”

Overall, this is a very strong Week 2 README; with the confusion-matrix and misclassified-gallery bug fixed, it will look like a polished mini paper.
