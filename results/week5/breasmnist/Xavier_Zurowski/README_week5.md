# Week 5 Core Deliverables — Method & Results

## Method Choice  
For week 5, my week4 SSL pretrained pt was used, and I began evaluating the representation quality with linear probing. The SSL encoder was fully frozen, and only a linear classification head was trained. 

In following with week4 baseline, the following data augmentations were applied during SSL pretraining:

```python
ssl_transform = vT.Compose([
    vT.RandomResizedCrop(224, scale=(0.7, 1.0)),
    vT.RandomHorizontalFlip(p=0.5),
    vT.RandomApply([vT.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    vT.RandomRotation(degrees=10),
    vT.RandomGrayscale(p=0.2),
    vT.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    vT.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225]),
```
## Experimental Setup

Linear probes were trained using 5% and 10% of labeled data, and evaluated on the test split. Performance was measured using accuracy, balanced accuracy (since BreastMNIST is unbalanced), and AUROC. Results were compared against my supervised baseline on 100% of the data. Best threshold was tuned on the validation split of BreastMNIST for balanced accuracy.

Setup for the linear probes:
```
Optimizer: Adam
Learning rate: 1e-3
Epochs: 25
Batch size: 128
Weight decay: None
Loss: CrossEntropyLoss

```

## Results
**Linear Probe Performance**
| Label Fraction | Accuracy | Balanced Accuracy | AUROC | Best Threshold|
| -------------- | -------- | ----------------- | ----- | -------------- |
| 5%             | 0.622   | 0.651             | 0.717 | 0.633 |
| 10%            | 0.686   | 0.650           | 0.726 | 0.523 |
| 20%            | 0.679   | 0.653           | 0.789 | 0.412 |

**Supervised Baselines**
| Model Type                         | Accuracy | Balanced Accuracy | AUROC |
| ---------------------------------- | -------- | ----------------- | ----- |
| Fully supervised (enc + head)      | 0.897    | 0.854             | 0.928 |


## Analysis  
The linear probe results indicate that the SSL-pretrained encoder learns features that are meaningfully descriminative even under severe label scarcity. With only 5–10% labeled data, AUROC remains above 0.71 and improves to 0.79 at 20%, suggesting that the representation captures some class-separable structure independent of the linear head. Notably, the balanced accuracy score remains relatively stable (~0.65) across 5–20%, despite increasing overall accuracy. This suggests that gains from additional labeled data primarily improve confidence calibration and threshold positioning rather than substantially improving minority-class discrimination. The rising AUROC with 20% labels further supports that the learned representation contains useful ranking information, even if the default decision boundary is suboptimal.

However, there remains a significant performance gap compared to the fully supervised model trained on 100% labels (AUROC 0.928 vs 0.789 at best). This gap highlights key limitations, firtly that freezing the encoder constrains adaptation to the downstream distribution, and secondly the linear head alone may lack sufficient capacity to exploit nonlinear separability in the feature space. Finaly, the results suggest that the SSL encoder provides a transferable starting point, but fine-tuning the encoder is likely necessary to approach fully supervised performance.

