## **Model 1**
```
ssl_transform = vT.Compose([
    vT.RandomResizedCrop(224, scale=(0.7, 1.0)),
    vT.RandomHorizontalFlip(p=0.5),
    vT.RandomApply([vT.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    vT.RandomRotation(degrees=10),
    vT.RandomGrayscale(p=0.2),
    vT.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    vT.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225]),
])
```
## **Model 2**
```
ssl_transform = vT.Compose([
    vT.RandomResizedCrop(224, scale=(0.7, 1.0)),
    vT.RandomHorizontalFlip(p=0.5),
    vT.RandomApply([vT.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
    vT.RandomRotation(degrees=10),
    vT.RandomGrayscale(p=0.1),
    vT.GaussianBlur(kernel_size=3, sigma=(0.1, 0.25)),
    vT.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225]),
])
```

## Results by Label Fraction

### 5% Labels

| Model   | Accuracy | Balanced Acc | AUROC |
|---------|----------|--------------|--------|
| Model 1 | 0.712    | 0.690        | 0.755  |
| Model 2 | 0.731    | 0.726        | 0.748  |
| Model 3 | 

---

### 10% Labels

| Model   | Accuracy | Balanced Acc | AUROC |
|---------|----------|--------------|--------|
| Model 1 | 0.686    | 0.702        | 0.781  |
| Model 2 | 0.705    | 0.716        | 0.782  |

---

### 20% Labels

| Model   | Accuracy | Balanced Acc | AUROC |
|---------|----------|--------------|--------|
| Model 1 | 0.776    | 0.779        | 0.831  |
| Model 2 | 0.750    | 0.739        | 0.783  |
