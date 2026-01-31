# Week 6 – SSL Finetuning and Calibration

## Overview

For week6 I explored fine-tuning a pre-trained self-supervised encoder from week5 on the breastMNIST data set with 10% and 20% labeled slpits. The goal was to assess how fine-tuning affects model performance and calibration, and to compare it with linear probing.  

I performed experiments with two fractions of labeled data:

* **10% labeled subset**
* **20% labeled subset**

I evaluated each model using:

* **Accuracy (acc)**  
* **Balanced Accuracy (balanced_acc)** – accounts for class imbalance  
* **AUROC (auroc)** – measures ranking performance  
* **Expected Calibration Error (ECE)** – measures how well predicted probabilities match true outcomes  

A reliability diagram was also produced for each finetuned model to visualize calibration.

## Method

### 1. Linear Probe

* The pre-trained SSL encoder was frozen, and a small linear head was trained on the labeled subset.  
* Training used cross-entropy loss with class weights to correct for imbalance.  
* The probe was evaluated on the full validation set.

### 2. Fine-Tuning

* The entire SSL encoder was unfrozen and trained jointly with the head.  
* Learning rates were separated: small LR for the encoder (`1e-5`) and larger LR for the head (`1e-3`).  
* Regularization included:
  * Dropout in the head (`p=0.5`)  
  * Weight decay (`1e-4`)  
  * Label smoothing (`0.1`)  

## Results

### 10% Labels

| Model       | Accuracy | Balanced Accuracy | AUROC  | ECE   |
|------------|----------|-----------------|--------|-------|
| SSL Probe  | 0.724    | 0.721           | 0.751  | –     |
| SSL Finetune | 0.718  | 0.596           | 0.683  | 0.102 |

**Observations:**

* Fine-tuning with only 10% of labels reduced AUROC and balanced accuracy when compared to the linear probe.  
* Accuracy remained similar, but balanced accuracy dropped significantly, suggesting overfitting to the majority class.  
* ECE decreased slightly (0.102), indicating the model’s confidence is better calibrated, despite worse discrimination.  
* Interpretation: With very limited labeled data, fine-tuning the entire encoder can overfit, especially to dominant classes, reducing ranking performance (AUROC) and balanced accuracy.

### 20% Labels

| Model       | Accuracy | Balanced Accuracy | AUROC  | ECE   |
|------------|----------|-----------------|--------|-------|
| SSL Probe  | 0.763    | 0.725           | 0.750  | –     |
| SSL Finetune | 0.788  | 0.697           | 0.760  | 0.122 |

**Observations:**

* With 20% labeled data, fine-tuning improves overall accuracy and AUROC compared to the probe.  
* Balanced accuracy is slightly lower than the probe, indicating minor overfitting to the majority class, but the model is better at ranking positive examples.  
* ECE increased slightly (0.122), suggesting a slight decrease in calibration, largely still acceptable.  
* Interpretation: More labeled data allows the encoder to adapt effectively without catastrophic overfitting, improving discriminative performance.

## Analysis

* **Effect of label fraction:**  
  * At **10% labels**, fine-tuning hurts AUROC and balanced accuracy due to overfitting to limited labeled samples.  
  * At **20% labels**, fine-tuning improves AUROC and accuracy, showing that the encoder can effectively adapt when sufficient labels are available.

* **Calibration (ECE):**  
  * Fine-tuning tends to slightly increase or decrease ECE depending on the amount of data.  
  * Reliability diagrams indicate that with low-data fine-tuning, probabilities are slightly better calibrated (10% trial), but at higher fractions (20%), calibration may slightly degrade due to overconfidence.

### Reliability Diagrams
**10% labeled data.**
The reliability diagram shows unstable calibration, especially in low-confidence bins, where accuracy is disproportionately high, indicating underconfidence driven by limited data and small bin counts. Mid-confidence bins are closer to the diagonal, while high-confidence bins show mild overconfidence. This aligns with the lower ECE but weaker AUROC and balanced accuracy.

**20% labeled data.**
With more labeled data, calibration becomes smoother and more consistent across bins. Mid–high confidence predictions better align with the diagonal, though the highest-confidence bins exhibit slight overconfidence. This corresponds to improved accuracy and AUROC but a modest increase in ECE.

## Conclusion

* **Linear probe** is robust for extremely limited labeled data (10%), outperforming full fine-tuning in AUROC and balanced accuracy.  
* **Fine-tuning** is beneficial as labeled data increases (20%), improving accuracy and AUROC while maintaining reasonable calibration.  




