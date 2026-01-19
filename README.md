# medmnist-ssl — Weeks 1–10 README (Research Plan + How-To)

> **Goal:** Undergraduate-friendly, **research-oriented** project using **MedMNIST2D (binary-class only)** with a required **Self-Supervised Contrastive Learning (SSL)** phase. We will build supervised baselines, pretrain with SSL, and evaluate **label efficiency** and **calibration** on two datasets.

* Site: [https://medmnist.com/](https://medmnist.com/)
* Primary datasets: **BreastMNIST** (`breastmnist`), **PneumoniaMNIST** (`pneumoniamnist`)
* Repo name: ``

We deliberately keep the **core pipeline shared** across datasets so we can compare results, but from **Week 5 onward** each dataset gets its own **extra focus tasks**.

---

## 0) Roles, Primary Datasets & Task Ownership

### 0.1 Primary dataset (must-have)

* Each member chooses **one primary dataset**:

  * `breastmnist` (breast ultrasound, benign vs malignant)
  * `pneumoniamnist` (chest X-ray, pneumonia vs normal)
* You will run **all core tasks (Weeks 1–4 and the "shared core" part of Weeks 5–7)** on your **primary dataset**.
* For fairness and clean comparisons, your **main plots/tables** in the final writeup are based on your **primary dataset**.

### 0.2 Helper / extra tasks (nice-to-have)

* From **Week 5 onward**, there are **extra tasks** that go *deeper* on each dataset:

  * **PneumoniaMNIST focus** → calibration, thresholds, robustness, operating points.
  * **BreastMNIST focus** → label efficiency, augmentations, representations.
* These extra tasks are managed via a **Task Board** (see below) and can be taken by:

  * people whose **primary dataset matches** the task, and
  * **volunteers from the other dataset** acting as *helpers*.
* Example:

  * If there are more Pneumonia people than Breast people, a few Pneumonia members can sign up for **Breast extra tasks** as helpers, while still keeping Pneumonia as their primary dataset.

### 0.3 How to claim a primary dataset

1. Create a GitHub Issue titled:

   * `Signup: <your_name> — primary=<dataset_key>`
2. In the issue, include:

   * system: Colab vs local (e.g., `Colab GPU`)
   * preferred model for fast runs: `smallcnn` vs `resnet18`
3. Once you pick a primary dataset, **keep it for all weeks**.

### 0.4 Task Board (Weeks 5–7)

From **Week 5 onward**, we avoid chaos and duplication by using a **Task Board**:

* File: `reports/task_board_week5_7.md`
* Each extra task has a row with:

```markdown
| Task | Dataset | Description                               | Owner 1 | Owner 2 |
|------|---------|-------------------------------------------|---------|---------|
| P1   | pneumo  | Label fractions 1/2/5/10/20% + repeats    |         |         |
| P2   | pneumo  | Temp scaling vs no-TS (ECE)               |         |         |
| ...  | ...     | ...                                       |         |         |
```

* Everyone must:

  * Do all **core tasks** on their primary dataset (no signup needed).
  * Sign up for **1–2 extra tasks** in the Task Board.
  * If team sizes are imbalanced (e.g., many Pneumonia vs few Breast), members of the larger team should take **1 helper task on the smaller team’s dataset**.

**When you take a task:**

* Create an Issue:

  * Title: `Week5 extra: <task_id> – <dataset> – <your_name>`
  * Restate the goal briefly
  * Link to your `results/weekX/<dataset>/<your_name>/` folder and PRs.

---

## 1) Repository Structure

```text
medmnist-ssl/
  ├─ notebooks/
  │   └─ ML_Basics_MedMNIST_Colab.ipynb
  ├─ starter/
  │   ├─ requirements.txt
  │   └─ src/
  │       ├─ data.py
  │       ├─ metrics.py
  │       ├─ models.py
  │       ├─ train_medmnist.py
  │       └─ utils.py
  ├─ results/
  │   ├─ week1/ ... week10/
  │   │   └─ <dataset_key>/<your_name>/
  ├─ reports/
  │   ├─ task_board_week5_7.md
  │   └─ final_writeup/
  ├─ figures/
  ├─ README.md
  └─ .gitignore
```

### .gitignore (minimum)

```text
runs/
__pycache__/
.ipynb_checkpoints/
*.pt
*.png
*.json
```

> Commit only curated artifacts to `results/` (metrics/plots/screens). Do **not** commit bulky raw `runs/`.

---

## 2) Environment & Tools

* **Primary runtime:** Google Colab (GPU). Local runs are optional but allowed.
* **Core packages:** `torch`, `torchvision`, `medmnist`, `scikit-learn`, `matplotlib`, `tqdm`, `numpy`.
* Data is downloaded automatically by `medmnist` to a cache (`~/.medmnist` or `/root/.medmnist` on Colab).

### 2.1 Colab quickstart

1. Open `notebooks/ML_Basics_MedMNIST_Colab.ipynb` in Colab.
2. Set **GPU**: Runtime → Change runtime type → GPU.
3. Install dependencies (if needed):

```python
!pip -q install medmnist torch torchvision scikit-learn matplotlib tqdm
```

4. In the config cell, set:

```python
DATASET_KEY = 'pneumoniamnist'  # or 'breastmnist'
MODEL_NAME  = 'resnet18'        # or 'smallcnn'
EPOCHS      = 5
BATCH_SIZE  = 128
```

### 2.2 Local quickstart (optional)

```bash
pip install -r starter/requirements.txt
python -m src.train_medmnist --dataset breastmnist --model smallcnn --epochs 5
# or
python -m src.train_medmnist --dataset pneumoniamnist --model resnet18 --finetune head --epochs 5
```

**Dataset auto-download example:**

```python
import medmnist
from medmnist import INFO

KEY = 'breastmnist'      # or 'pneumoniamnist'
DataClass = getattr(medmnist, INFO[KEY]['python_class'])

train_ds = DataClass(split='train', download=True)
val_ds   = DataClass(split='val',   download=True)
test_ds  = DataClass(split='test',  download=True)

print('Cache root:', train_ds.root)
print('Shape:', train_ds.imgs.shape)
print('Labels:', INFO[KEY]['label'])
```

---

## 3) Week-by-Week Plan (Overview)

We keep a **shared pipeline** across datasets but allow **dataset-specific extra tasks** from Week 5 onward.

* **Week 1:** Kickoff & single baseline run (binary only)
* **Week 2:** EDA, baselines & first calibration snapshot
* **Week 3:** Finetune-all & light augmentations
* **Week 4:** SSL pretraining (SimCLR-lite / MoCo-lite)
* **Week 5:** Linear probe & label efficiency (shared core + extra tasks)
* **Week 6:** SSL fine-tuning & calibration (shared core + extra tasks)
* **Week 7:** Thresholds, PR/ROC, error analysis (shared core + extra tasks)
* **Week 8:** Ethics, limits, robustness mini-test
* **Week 9:** Writing & figures
* **Week 10:** Final talk & reproducibility

Below: details per week.

---

### Week 1 — Kickoff & Baseline Run (Binary Only)

**Everyone (on their primary dataset) must:**

* Pick `breastmnist` or `pneumoniamnist` as **primary** (via signup Issue).
* Run a 5-epoch baseline (`smallcnn` or `resnet18 --finetune head`).
* Check outputs: `best.pt`, `val_log.jsonl`, `test_metrics.json`, `reliability.png`.

**Expected folder:**

```text
results/week1/<dataset>/<your_name>/
  ├─ test_metrics.json
  ├─ reliability.png
  ├─ screenshot_env.png
  └─ README_week1.md
```

`README_week1.md` (3–5 sentences):

* model / epochs / batch size
* one bug you hit + how you fixed it
* note on Acc / AUROC / (ECE if available)

**Quick intuition:**

* **BreastMNIST** often shows **lower AUROC** but sometimes better calibration.
* **PneumoniaMNIST** often has **high AUROC** but tends to be **overconfident** (high ECE).

These patterns will be revisited in later weeks.

---

### Week 2 — EDA, Baselines & First Calibration Check

> **Theme:** "Know your data, trust your baselines."

Goals (per person, on primary dataset):

1. Understand **basic dataset statistics**:

   * Class counts (train/val/test)
   * Example images per class
   * Input shape / value range
2. Train **two supervised baselines**:

   * `smallcnn`
   * `resnet18 --finetune head` (backbone frozen)
3. Produce a **short comparison**:

   * Accuracy & AUROC for both models
   * 3–5 misclassified examples (gallery)
   * First **calibration snapshot** (reliability diagram + ECE)

**Expected folder:**

```text
results/week2/<dataset>/<your_name>/
  ├─ metrics_smallcnn.json
  ├─ metrics_resnet18_head.json
  ├─ cm_smallcnn.png
  ├─ cm_resnet18_head.png
  ├─ misclassified_gallery.png
  ├─ reliability_smallcnn.png        # optional
  ├─ reliability_resnet18_head.png
  └─ README_week2.md
```

`README_week2.md` (0.5–1 page) should contain:

1. **Dataset recap:** class balance, artifacts, obvious quirks.
2. **Baseline comparison:** table + comments.
3. **Calibration snapshot:** ECE and 3–5 sentences of interpretation.
4. **Error analysis:** bullet list of what misclassified examples suggest.

This week defines the **supervised reference point** for Weeks 3–7.

---

### Week 3 — Finetune-All & Light Augmentations

> **Theme:** "Same backbone, different training regimes."

Everyone (on their primary dataset) should:

1. Compare two ResNet-18 regimes:

   * **Head-only** (frozen backbone, Week 2 baseline)
   * **Finetune-all** (all layers trainable)
2. Introduce **light, medically reasonable augmentations** on the **train** split.
3. Read **learning curves** and diagnose over/underfitting.
4. Decide on a **recommended supervised baseline config** to use going into SSL.

**Expected folder:**

```text
results/week3/<dataset>/<your_name>/
  ├─ metrics_resnet18_head.json
  ├─ metrics_resnet18_all_basic.json
  ├─ metrics_resnet18_all_augA.json
  ├─ metrics_resnet18_all_augB.json      # optional
  ├─ curves_resnet18_head.png
  ├─ curves_resnet18_all_basic.png
  ├─ curves_resnet18_all_augA.png
  ├─ aug_ablation_table.md (optional)
  └─ README_week3.md
```

#### Suggested augmentations

* **PneumoniaMNIST:**

  * `RandomHorizontalFlip(p=0.5)`
  * `RandomRotation(degrees=10)` or small `RandomAffine`
  * Optional: small `ColorJitter(brightness=0.1, contrast=0.1)`
  * Avoid vertical flips / huge rotations.

* **BreastMNIST:**

  * `RandomHorizontalFlip(p=0.5)`
  * `RandomRotation(degrees=10)`
  * Optional: light Gaussian noise or tiny contrast jitter.

`README_week3.md` (0.5–1.5 pages) should have:

1. Setup recap and transforms used.
2. **Head-only vs finetune-all** mini-table + discussion.
3. **Augmentation ablation** table + commentary.
4. Learning curve interpretation.
5. Takeaways: which config becomes your **reference supervised baseline**.

---

### Week 4 — SSL Pretraining (Required, Shared)

> **Theme:** "Learn good representations from unlabeled images before spending labels."

In Week 4, **everyone** implements a *minimal but real* SSL method on their **primary dataset**. We treat the **train split as unlabeled**, generate two augmented views per image, and train an encoder with a **contrastive loss**. This encoder will then be reused in Weeks 5–7.

We keep things as simple and reproducible as possible: the **default** is a small **SimCLR-lite** implementation; **MoCo-lite** is an **optional advanced track** for anyone who wants to go further.

---

#### 4.0 Roadmap (what you actually do this week)

By the end of Week 4, you should have done:

1. **Create an SSL training entry point**

   * Option A (notebook, recommended for most):

     * Copy `notebooks/ML_Basics_MedMNIST_Colab.ipynb` → `notebooks/Week4_SSL_<your_name>.ipynb`.
   * Option B (script, recommended if you like CLIs):

     * Create `starter/src/train_ssl.py` (you can copy structure from `train_medmnist.py`).

2. **Build an SSL DataLoader**

   * Reuse your MedMNIST `train` split from Week 2–3.
   * Wrap it in an `SSLDataset` that returns **two augmented views** per image.

3. **Define encoder + projection head**

   * Use **ResNet-18 backbone** (same as supervised, without the final classifier).
   * Add a small 2-layer projection MLP.

4. **Implement SimCLR-lite NT-Xent loss**

   * Write a function `nt_xent_loss(z1, z2, tau)`.
   * It should treat each pair `(z1[i], z2[i])` as positives and everything else in the batch as negatives.

5. **Train for ~30 epochs**

   * Log loss per epoch.
   * Plot `ssl_loss_curve.png` (loss vs epoch).

6. **Save encoder + config**

   * Save encoder weights (e.g., `ssl_encoder.pt`).
   * Save `ssl_config.json` with all key hyperparameters & augmentations.

7. **Write `README_week4.md`**

   * 0.5–1 page: method choice, augmentations, hyperparams, behavior of loss, notes for Week 5.

If you want a challenge, you can implement **MoCo-lite** instead of SimCLR-lite, but the **core requirement** is that you have:

* a working SSL encoder,
* a loss curve,
* and a clear configuration file.

---

#### 4.1 Goals (formal)

For **your primary dataset**, by the end of Week 4 you must have:

1. A working **SSL training loop** that:

   * takes the train split as *unlabeled* images;
   * applies **strong augmentations** twice per image;
   * computes a **contrastive loss** (SimCLR-lite or MoCo-lite) and backpropagates.
2. A trained **SSL encoder checkpoint** (e.g., `ssl_encoder.pt`).
3. A **loss vs epoch curve** (`ssl_loss_curve.png`).
4. A short **`README_week4.md`** documenting your design choices and any issues.

This does **not** need to be a large-scale SimCLR. A clean, well-documented small version is enough.

---

#### 4.2 Architecture: encoder & projection head

We reuse the supervised **ResNet-18** backbone from Weeks 2–3.

* **Base encoder**: ResNet-18 up to the global average pooling layer.
* **Projection head**: small MLP, e.g.:

```python
# Example projection head
proj_head = nn.Sequential(
    nn.Linear(feat_dim, proj_dim),  # e.g. 512 -> 128
    nn.ReLU(inplace=True),
    nn.Linear(proj_dim, proj_dim),  # 128 -> 128
)
```

* During SSL training:

  * We feed two views `x1`, `x2` of the **same image**.
  * Encoder + projector produce `z1`, `z2`.
  * We **L2-normalize** `z1`, `z2` along the feature dimension.
  * We apply the contrastive loss so that:

    * `z1` is close to `z2` (positive pair),
    * `z1` is far from all other embeddings in the batch (negatives).

**Important:**

* In Week 4, we **do not** use labels in the loss.
* In Weeks 5–6, we will reuse the encoder and **discard/rebuild** the projection head for downstream tasks.

---

#### 4.3 Loss: SimCLR-lite NT-Xent (default)

We recommend a **SimCLR-like NT-Xent loss** because it is simple and works with a single encoder.

Let `B` be the batch size and `z1, z2` be `[B, D]` matrices of **normalized** embeddings.

Sketch (conceptual, not copy-paste):

```python
def nt_xent_loss(z1, z2, tau=0.2):
    # z1, z2: [B, D], already L2-normalized
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)    # [2B, D]

    # cosine similarities
    sim = z @ z.T / tau               # [2B, 2B]

    # mask self-similarity
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: for index i, the positive is i+B (mod 2B)
    labels = torch.arange(2*B, device=z.device)
    labels = (labels + B) % (2*B)

    loss = F.cross_entropy(sim, labels)
    return loss
```

Key points:

* `tau` (temperature) controls how sharp the softmax is (we suggest **0.2**).
* We treat each embedding as a query that must identify its matched view among all 2B-1 candidates.

If you implement **MoCo-lite** instead:

* You need a **momentum encoder** and a **queue** of negative keys.
* Document `queue_size` (e.g. 2048–8192) and `momentum` (e.g. 0.99) in `ssl_config.json`.

---

#### 4.4 Augmentations for SSL (stronger than Week 3)

For SSL we use **strong but still medically reasonable** augmentations. Define a dedicated SSL transform (separate from the supervised `train_transform`).

Example pattern:

```python
ssl_transform_pneumo = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    # Normalize(...) same as supervised
])
```

Dataset-specific hints:

* **PneumoniaMNIST (chest X-ray)**

  * OK: horizontal flips, small rotations, small translations, mild contrast/brightness jitter.
  * Avoid: vertical flips, very large rotations (≥30°), color jitter that inverts or destroys anatomy.

* **BreastMNIST (ultrasound)**

  * OK: horizontal flips, small rotations, random resized crops, light Gaussian noise.
  * Use contrast changes carefully; ultrasound is already noisy.

**SSL dataset wrapper**

You should create a wrapper that returns **two views** per sample:

```python
class SSLDataset(Dataset):
    def __init__(self, base_dataset, ssl_transform):
        self.base = base_dataset
        self.transform = ssl_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]      # ignore label
        v1 = self.transform(x)
        v2 = self.transform(x)
        return v1, v2
```

Then build a DataLoader:

```python
ssl_train_ds = SSLDataset(train_ds, ssl_transform)
ssl_train_loader = DataLoader(
    ssl_train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
```

---

#### 4.5 Training schedule & hyperparameters

Suggested defaults (tune if needed, but **log everything** in `ssl_config.json`):

* **Backbone:** ResNet-18
* **Projection dim:** 128
* **Batch size:** 128 (64 if GPU memory is tight)
* **Epochs:** 30 (up to 50 if you have time)
* **Optimizer:** AdamW

  * lr: 1e-3 (if unstable, reduce to 3e-4)
  * weight_decay: 1e-4
* **Temperature τ:** 0.2
* **Seed:** fix e.g. `seed=42` and document it.

Sketch of a training loop:

```python
for epoch in range(1, num_epochs + 1):
    encoder.train(); proj_head.train()
    running_loss = 0.0

    for (v1, v2) in ssl_train_loader:
        v1, v2 = v1.to(device), v2.to(device)

        h1 = encoder(v1)           # [B, feat_dim]
        h2 = encoder(v2)
        z1 = proj_head(h1)
        z2 = proj_head(h2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss = nt_xent_loss(z1, z2, tau=tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * v1.size(0)

    epoch_loss = running_loss / len(ssl_train_loader.dataset)
    print(f"[SSL] Epoch {epoch}/{num_epochs} - loss={epoch_loss:.4f}")
    history.append(epoch_loss)

# After loop: plot history to ssl_loss_curve.png
```

(You can implement this either inside a notebook or inside `train_ssl.py`.)

---

#### 4.6 Expected artifacts

For Week 4, we expect **at least** the following files:

```text
results/week4/<dataset>/<your_name>/
  ├─ ssl_config.json            # hyperparameters & augmentations
  ├─ ssl_loss_curve.png         # loss vs epoch
  ├─ ssl_encoder.pt             # encoder weights (optional but recommended)
  ├─ ssl_proj_head.pt           # projection head (optional but useful)
  └─ README_week4.md
```

`ssl_config.json` should include (example):

```json
{
  "method": "simclr-lite",
  "backbone": "resnet18",
  "proj_dim": 128,
  "batch_size": 128,
  "epochs": 30,
  "optimizer": "AdamW",
  "lr": 0.001,
  "weight_decay": 0.0001,
  "tau": 0.2,
  "augmentations": "RandomResizedCrop+Flip+Rot+Jitter (see code)",
  "dataset": "pneumoniamnist",
  "seed": 42
}
```

`README_week4.md` (~0.5–1 page) template:

1. **Method choice**

   * SimCLR-lite (or MoCo-lite) and why you picked it.
2. **Augmentations**

   * 2–3 sentences describing your SSL transform and why it is realistic for your modality.
3. **Hyperparameters**

   * Batch size, epochs, τ, optimizer, anything special.
4. **Training behavior**

   * How the loss evolved (e.g., "decreased quickly then plateaued around 0.4").
   * Any instability or tricks you had to add (e.g., lower lr).
5. **Notes for Week 5–6**

   * Where the encoder checkpoint lives.
   * Any suspicion that it is undertrained/overtrained.

---

#### 4.7 Dataset-specific notes (optional but recommended)

If you want a small dataset-specific section inside `README_week4.md`:

* **PneumoniaMNIST:**

  * Comment on whether stronger augmentations (e.g., bigger crops/rotations) helped the SSL loss or made optimization unstable.
  * Note if representations *seem* to cluster by label when you briefly test a linear classifier on 100% labels (optional sanity check).

* **BreastMNIST:**

  * Discuss how sensitive the SSL training seemed to contrast/noise augmentations.
  * Mention if very strong augmentations appear to erase small lesions (visual check).

This text can be short (3–6 sentences per dataset) but will help interpret Week 5–6 results.

---

#### 4.8 Week 4 Checklist

You are ready to move on to **Week 5** when:

* [ ] You have a working **SSL training loop** that runs for at least ~20–30 epochs without crashing.
* [ ] You saved **`ssl_encoder.pt`** (and optionally `ssl_proj_head.pt`) for your primary dataset.
* [ ] You generated **`ssl_loss_curve.png`** and can describe how the loss behaves over epochs.
* [ ] You wrote **`ssl_config.json`** with all important hyperparameters and augmentations.
* [ ] You wrote **`README_week4.md`** summarizing method, augs, hyperparams, and any issues.

---

## 4) Weeks 5–7: Shared Core vs Extra Tasks: Shared Core vs Extra Tasks

From **Week 5 onward** we split the work into:

1. **Shared core tasks** (everyone does these on their primary dataset).
2. **Extra tasks** (dataset-specific focus, selected via Task Board).

This allows us to:

* Keep a **clean, comparable core** across Breast/Pneumonia.
* Avoid duplicated work in deeper analyses.
* Let people from the larger team help the smaller team.

> **Action item at the start of Week 5:**
>
> * Open `reports/task_board_week5_7.md`.
> * Put your name under **1–2 extra tasks** (P-tasks for Pneumonia focus, B-tasks for Breast focus).
> * If your dataset team is larger, plan for **at least one helper task** on the smaller dataset so that work is balanced.

### 4.1 Shared Core Tasks (must-do, per person)

For **each person (on their primary dataset)**:

* **Week 5 core:**

  * Freeze SSL encoder.
  * Train **linear probes** at **5% and 10%** label fractions.
  * Compare with supervised-only baseline.

* **Week 6 core:**

  * Unfreeze SSL encoder (small LR) for **fine-tuning**.
  * Compare **baseline vs SSL-probe vs SSL-finetune** in terms of Acc, AUROC, ECE.
  * Produce at least **one reliability diagram** for an SSL model.

* **Week 7 core:**

  * Choose at least **one operating point**:

    * e.g., threshold for target TPR (0.9) or target FPR (0.1).
  * Report metrics at that operating point.
  * Collect a small set of **high-confidence errors** with commentary.

These core tasks ensure we can always compare Breast vs Pneumonia in a unified way.

### 4.2 Extra Tasks — PneumoniaMNIST Focus

For PneumoniaMNIST (chest X-ray), our extra tasks focus on **calibration, thresholds, and robustness**. Examples of extra task IDs (for Task Board):

* **P1:** Finer label fractions (1%, 2%, 5%, 10%, 20%) + repeated runs (variance).
* **P2:** Compare **temperature scaling / Platt scaling** vs no calibration (ECE before/after).
* **P3:** Mini **domain shift** experiments (e.g., brightness/contrast/noise) and measure AUROC/ECE change.
* **P4:** ROC/PR analysis of **trade-off** between "missing pneumonia" (FN) vs "over-calling pneumonia" (FP).
* **P5:** High-confidence FP/FN **error gallery** with qualitative discussion.

### 4.3 Extra Tasks — BreastMNIST Focus

For BreastMNIST (breast ultrasound), extra tasks emphasize **label efficiency, augmentation, and representations**:

* **B1:** Label efficiency grid (1%, 5%, 10%) with **2–3 random seeds** to estimate variance.
* **B2:** t-SNE/UMAP **embedding visualizations** comparing supervised vs SSL representations.
* **B3:** SSL **augmentation config** ablation (AugA vs AugB) and effects on downstream linear probes.
* **B4:** "Hard-case" error gallery for **small lesions / low-contrast cases**, with hypotheses.
* **B5:** Breast-specific **SSL vs no-SSL summary figure/table** for the final report.

Each of these tasks should have **1–2 owners** in `reports/task_board_week5_7.md`.

---

## 4) Weeks 5–7 — Two-Week Sprint (Teamwork + Individual Tasks + References)

> **Theme:** “From SSL to decisions.” We compress Weeks **5–7** into a **2-week sprint**. We keep a **shared core** so results are comparable, but we also assign **dataset-specific extra tasks** via a Task Board to avoid duplicated work.

### 4.0 Sprint timeline (2 weeks total)

* **Sprint Week 1 (Probe + Fine-tune + Calibration)**

  * Linear probe with limited labels (**label efficiency**)
  * SSL fine-tuning (small LR)
  * Calibration evaluation (**ECE + reliability diagram**)

* **Sprint Week 2 (Threshold + ROC/PR + Error Analysis)**

  * Choose operating point(s) (**threshold selection**)
  * ROC / PR curves
  * High-confidence errors (FP/FN) + short qualitative notes

> **Important:** Starting from Week 5, we provide **pseudocode + required outputs**, but **not full starter code**. You are expected to implement the details by reusing your Week 3–4 pipeline.

---

### 4.1 Shared Core (everyone must do this on their primary dataset)

Each person completes the **core** on their **primary dataset** so we can compare Breast vs Pneumonia cleanly.

**Core deliverables (Sprint Week 1):**

1. **Label fractions:** run at least **5% and 10%** labels
2. **Linear probe:** freeze SSL encoder → train linear head
3. **Fine-tune:** unfreeze SSL encoder → small LR fine-tune (at least at **10%**)
4. **Calibration:** compute **ECE** + at least **one reliability diagram** (recommended: for the fine-tuned SSL model)
5. **Comparison table:** supervised baseline (Week 3 best) vs SSL probe vs SSL finetune (**Acc/AUROC/ECE**)

**Core deliverables (Sprint Week 2):**

1. **ROC curve** + **PR curve** (positive class)
2. **Operating point:** pick a threshold using the **validation** set (e.g., target **TPR≈0.90** or **FPR≈0.10**) and report metrics at that operating point
3. **High-confidence error gallery:** ~5 FP + ~5 FN with predicted probability and 1-line commentary

---

### 4.2 Dataset-specific extra tasks (Weeks 5–7)

To balance workloads and avoid duplication, extra tasks are split by dataset. **Everyone signs up for 1–2 extra tasks** via the Task Board:

* File: `reports/task_board_week5_7.md`
* Each task has:

  * **Owner 1 (Lead):** produces final artifacts + short writeup
  * **Owner 2 (Reviewer/Runner):** sanity checks results, helps reproduce/re-run, reviews interpretation

> If team sizes are imbalanced (e.g., Pneumonia has more members), members of the larger team should take **at least one helper task** on the smaller dataset.

#### 4.2.1 PneumoniaMNIST extra tasks (P-tasks)

Focus: **operating points, calibration, robustness, clinical trade-offs.**

* **P1 — Temperature scaling vs no calibration (ECE)**

  * Fit temperature on **val logits**, report ECE before/after; AUROC should stay ~same.
  * **DoD:** `P1_temp_scaling_table.md` + `P1_reliability_before.png` + `P1_reliability_after.png`.

* **P2 — Operating points: choose thresholds for target TPR/FPR**

  * Example targets: TPR=0.90, FPR=0.10; pick thresholds on val, report test metrics.
  * **DoD:** `P2_operating_points.md` + confusion matrices at each point.

* **P3 — PR vs ROC interpretation under imbalance**

  * Compare PR-AUC and ROC-AUC; write 8–12 lines interpreting why PR may be more informative.
  * **DoD:** `P3_pr_vs_roc_note.md` + `pr_curve.png` + `roc_curve.png`.

* **P4 — High-confidence error taxonomy (FP vs FN)**

  * Collect top-5 FP and top-5 FN by confidence; categorize (e.g., low-contrast, borderline, artifacts).
  * **DoD:** `P4_error_gallery.png` + `P4_error_notes.md`.

* **P5 — Mini robustness shift (tiny, cheap)**

  * Evaluate model under a mild perturbation (e.g., +Gaussian noise, small contrast shift, mild crop) and report ∆Acc/∆AUROC/∆ECE.
  * **DoD:** `P5_robustness_table.md` + 1 plot comparing metrics.

* **P6 (optional) — Calibration under shift**

  * Repeat ECE/reliability after the perturbation and discuss calibration drift.
  * **DoD:** `P6_shift_reliability.png` + 6–10 line discussion.

#### 4.2.2 BreastMNIST extra tasks (B-tasks)

Focus: **label efficiency, variability across seeds, augmentation sensitivity, representation quality.**

* **B1 — Label fraction grid + repeats (variance matters)**

  * Run fractions: 1/2/5/10/20% with **2–3 seeds** for linear probe; report mean±std.
  * **DoD:** `B1_label_grid.csv` + `B1_label_efficiency_plot.png` + short interpretation.

* **B2 — Augmentation ablation for SSL (representation quality)**

  * Compare 2 SSL augmentation configs (mild vs slightly stronger) and evaluate via linear probe (e.g., 10%).
  * **DoD:** `B2_aug_ablation_table.md` + 1 figure showing probe performance.

* **B3 — SSL vs ImageNet-init vs supervised-only (small-data focus)**

  * Compare three regimes at 10% labels: (i) SSL-pretrained, (ii) ImageNet init, (iii) supervised-from-scratch.
  * **DoD:** `B3_comparison_table.md` + 6–10 line discussion.

* **B4 — Representation visualization (optional)**

  * Use UMAP/t-SNE on encoder features (val or train subset) and comment on class separation.
  * **DoD:** `B4_umap.png` (or `B4_tsne.png`) + `B4_notes.md`.

* **B5 — Hard-case error patterns**

  * For high-confidence errors, identify recurring failure modes (speckle noise, weak boundaries, tiny lesions).
  * **DoD:** `B5_error_gallery.png` + bullets describing patterns and possible fixes.

---

### 4.3 References requirement (new)

From Week 5 onward, tasks involve reading and connecting to relevant references.

**For every extra task you own (Lead), you must:**

1. Find **≥1 relevant reference** (paper, tutorial, or high-quality note) that supports your method/analysis.
2. Write a short summary (8–12 lines) answering:

   * What is the key idea?
   * What protocol/metric does it recommend?
   * How are we using it in our MedMNIST setup?
3. Save it to:

```text
reports/references/<task_id>_<shortname>.md
```

Reviewer adds a short “sanity check” comment (3–5 lines) if possible.

---

### 4.4 Pseudocode pack (what we provide instead of starter code)

We provide **pseudocode** and required outputs, but you implement the code yourself using your Week 3–4 pipeline.

#### 4.4.1 Label fraction sampling (reproducible)

```text
function make_split_indices(train_dataset, frac, seed):
    set_seed(seed)
    y = labels(train_dataset)
    idx = stratified_sample(y, fraction=frac, seed=seed)
    save_json(indices_frac={frac}_seed={seed}.json, idx)
    return idx
```

#### 4.4.2 Linear probe (freeze encoder)

```text
function train_linear_probe(ssl_encoder, train_loader_frac, val_loader, cfg):
    freeze(ssl_encoder)
    head = Linear(feat_dim -> num_classes)
    train head with CE loss
    evaluate on val/test -> acc, auroc, probs
    save probe_metrics_{frac}.json
```

#### 4.4.3 Fine-tuning (unfreeze + small LR)

```text
function finetune(ssl_encoder, head, train_loader_frac, val_loader, cfg):
    unfreeze(ssl_encoder)
    set lr_encoder = small_lr, lr_head = larger_lr
    train encoder+head with CE loss
    evaluate on val/test -> acc, auroc, probs
    save finetune_metrics_{frac}.json
```

#### 4.4.4 Calibration (ECE + reliability)

```text
function compute_reliability(probs, y, n_bins):
    for each bin:
        conf_bin = mean(confidence)
        acc_bin  = mean(correct)
        count_bin = size
    ECE = sum (count_bin/N) * |acc_bin - conf_bin|
    plot reliability diagram + histogram
    return ECE
```

#### 4.4.5 Threshold, ROC/PR, high-confidence errors

```text
function choose_threshold(scores, y_val, target_metric, target_value):
    scan thresholds
    pick t that best matches target

function collect_high_conf_errors(scores, y, t, k):
    yhat = scores >= t
    FP = top-k scores among (yhat=1, y=0)
    FN = bottom-k scores among (yhat=0, y=1)
    save gallery/table with (image, score, true/pred, note)
```

---

### 4.5 Expected artifacts and folders (Weeks 5–7)

Use:

```text
results/week5/<dataset>/<your_name>/
results/week6/<dataset>/<your_name>/
results/week7/<dataset>/<your_name>/
```

**Minimum expected artifacts:**

* **Week 5 (Sprint Week 1):**

  * `probe_metrics_5pct.json`, `probe_metrics_10pct.json`
  * `probe_vs_supervised_plot_acc.png`, `probe_vs_supervised_plot_auroc.png`
  * `README_week5_core.md`

* **Week 6 (Sprint Week 1):**

  * `metrics_supervised_baseline.json`, `metrics_ssl_probe.json`, `metrics_ssl_finetune.json`
  * `reliability_ssl_finetune.png` (and ECE value recorded)
  * `README_week6_core.md`

* **Week 7 (Sprint Week 2):**

  * `roc_curve.png`, `pr_curve.png`
  * `operating_point_table.md`
  * `high_conf_error_gallery.png` (or table inside README)
  * `README_week7_core.md`

Extra task artifacts should be named clearly with the task ID, e.g. `P1_temp_scaling_ece.json` or `B1_label_grid_seeds.csv`.

---

### Week 8 — Ethics, Limits, and Mini Robustness Test

#### Recommended Reading for Weeks 5–7 (Calibration, ROC/PR, Thresholds)

These are **reference materials** for:

* calibration & reliability diagrams (Week 6),

* ROC/PR curves & operating points (Week 7),

* understanding how people evaluate models in imbalanced medical settings.

* **Calibration – classic paper**
  Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, *"On Calibration of Modern Neural Networks"*, ICML 2017.
  → Defines ECE, reliability diagrams, and temperature scaling; shows modern nets are often over-confident.

* **Calibration – modern follow-up (optional)**
  Minderer et al., *"Revisiting the Calibration of Modern Neural Networks"*, NeurIPS / ICLR-era work.
  → Looks at how calibration behaves in large-scale, modern architectures and under distribution shift.

* **ROC basics (very useful for medical ML)**
  Tom Fawcett, *"An Introduction to ROC Analysis"*, Pattern Recognition Letters, 2006.
  → Clear tutorial on ROC curves, AUC, and common pitfalls; good background for ROC plots in Week 7.

* **ROC vs Precision–Recall**
  Jesse Davis, Mark Goadrich, *"The Relationship Between Precision-Recall and ROC Curves"*, ICML 2006.
  → Explains why PR curves can be more informative than ROC when classes are imbalanced (like MedMNIST).

* **(Optional) Practical calibration notes**
  Lecture notes / blog posts that summarize ECE, reliability diagrams, and Platt/temperature scaling in a more applied way.

When you write Week 6–7 READMEs, you can:

* cite Guo et al. when you mention ECE / temperature scaling;
* use Fawcett + Davis & Goadrich for ROC/PR plots and threshold discussion.

---

### Week 8 — Ethics, Limits, and Mini Robustness Test

> **Theme:** "How and where does this fail?"

Here, tasks are mostly **shared across datasets**, but you can coordinate via the Task Board if needed.

**Everyone** should write a ~1-page note (`reports/week8/<your_name>_ethics_limits.md`) covering:

* Intended vs non-intended use of a simple MedMNIST classifier.
* Risks of overfitting, dataset bias, and spurious correlations.
* How calibration and operating points impact clinical risk.

Optional mini robustness test (per dataset or per team):

* Simple distribution shifts:

  * e.g., Gaussian noise, contrast change, small cropping.
* Measure how Acc / AUROC / ECE change.
* Summarize in a small table or plot.

---

### Week 9 — Writing & Figures

> **Theme:** "Turn experiments into a small paper."

As a group, start drafting a **4–6 page paper-style writeup**:

* **Intro:** motivation (label efficiency, SSL, calibration, two datasets).
* **Methods:** models, MedMNIST, SSL setup, evaluation metrics.
* **Experiments:** summarize Weeks 1–7 results.
* **Results:** key tables/plots.
* **Discussion:** what worked, where SSL helps, where it doesn’t.

Artifacts:

* `reports/final_writeup/draft_v1.pdf`
* All source figures saved under `figures/` with clear filenames.
* A small checklist of missing pieces (e.g., extra experiments still running).

---

### Week 10 — Talk & Final Artifacts

> **Theme:** "Communicate clearly and make it reproducible."

By the end of Week 10, you should have:

1. A **10–12 minute talk** (slides) summarizing:

   * Problem and datasets
   * Supervised baselines
   * SSL pretraining & label efficiency
   * Calibration and thresholds
   * Dataset-specific insights (Breast vs Pneumonia)
2. A **final PDF** of the writeup.
3. A **reproducibility checklist**, e.g.:

   * A single top-level README that explains how to:

     * Run a core baseline.
     * Run SSL pretraining.
     * Reproduce key plots on Colab in ≤ 1 hour.

Artifacts:

```text
results/week10/
  ├─ slides/<talk_title>.pdf
  ├─ final_writeup.pdf
  └─ reproducibility_checklist.md
```

---

## 5) Metrics & Evaluation

* **Primary metrics:**

  * Accuracy
  * **Macro-AUROC** (binary → AUROC)
  * **ECE** (Expected Calibration Error) with reliability diagrams
* **Secondary metrics:**

  * Precision, Recall, F1, confusion matrix

Always:

* Fix random seeds when possible.
* Log label fractions and hyperparameters.
* Report whether numbers are from **single run** or **mean over multiple seeds**.

---

## 6) Contribution Workflow

* Use GitHub **Issues** for questions and task signups:

  * `WeekX: <name> <dataset>` for questions
  * `Week5 extra: <task_id> – <dataset> – <your_name>` for extra tasks
* Use descriptive commit messages:

  * `Week3: finetune-all + aug ablation (breastmnist)`
* Prefer **PRs** for bigger changes; include:

  * short summary
  * screenshots of key plots (if relevant)

---

## 7) Troubleshooting (Colab)

* No GPU → Runtime → Change runtime type → GPU; reinstall packages.
* Too slow → switch to `smallcnn`, reduce `BATCH_SIZE`, or fewer epochs.
* Import errors → rerun install cell (Colab may reset environment).
* Save errors → check that `results/weekX/...` directory exists before writing.

---

## 8) Ethics, Privacy, Integrity

* Use only public MedMNIST data under its license.
* Never claim real clinical performance; this is an educational project.
* Report failures honestly (bad runs, unstable training, etc.).
* Do not cherry-pick only the best numbers without context.

---

## 9) Success Criteria

We consider the project successful if:

* We have **clean supervised and SSL baselines** on both BreastMNIST and PneumoniaMNIST.
* We can clearly answer:

  * When does SSL **improve label efficiency**?
  * How does SSL impact **calibration** and thresholds?
  * How do these effects differ between **Breast** and **Pneumonia**?
* The repo is tidy and **reproducible**, and we have:

  * A readable writeup.
  * A coherent talk.
  * Task Board and Issues that document who did what.
