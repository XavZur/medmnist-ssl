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
| 5%             | 0.583   | 0.632             | 0.707 | 0.618 |
| 10%            | 0.647   | 0.668           | 0.720 | 0.608 |
| 20%            | 0.730   | 0.710           | 0.782 | 0.452 |

**Supervised Baselines**
| Model Type                         | Accuracy | Balanced Accuracy | AUROC |
| ---------------------------------- | -------- | ----------------- | ----- |
| Fully supervised (enc + head)      | 0.897    | 0.854             | 0.928 |


## Analysis  
The linear probe results suggest that the SSL-pretrained encoder learns representations containing useful class information even with limited labeled data. As the labeled fraction increases from 5% to 20%, accuracy and AUROC both improve, indicating that the linear classifier can better separate the learned features with additional supervision. AUROC rises from 0.707 to 0.775, showing the encoder captures meaningful ranking information between classes despite being trained without labels. Balanced accuracy improves slightly, reflecting modest gains across both classes despite the dataset imbalance.

However, a substantial gap remains between the SSL probe and the fully supervised baseline. The supervised model achieves much higher performance, which is expected since its encoder is trained directly for the classification task. Because the linear probe restricts learning to a simple linear boundary in a frozen feature space, it cannot fully adapt to the dataset. These results suggest the SSL encoder provides a useful representation, but that further improvements likely require fine-tuning the encoder.

