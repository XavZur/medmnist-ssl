# medmnist-ssl — Week 1 README (GitHub + Colab)

> **Scope for Week 1**: We will run one end-to-end training on **MedMNIST2D (binary-class datasets only)**, save outputs, and push minimal artifacts to GitHub. Each team member **must pick exactly one binary dataset** and will be assigned to that dataset for the rest of the project.

* MedMNIST site: [https://medmnist.com/](https://medmnist.com/)
* This repo name: **`medmnist-ssl`** (private recommended)
* Audience: U0–U1 (McGill COMP202/250/206 level)

---

## 1) Binary Datasets (MedMNIST2D only)

We will restrict Week 1 to **binary** classification datasets in MedMNIST2D:

* **BreastMNIST** — benign vs malignant (binary)
* **PneumoniaMNIST** — pneumonia vs normal (binary)

> ✅ Pick **one** of the above. Each member will be **assigned** to the dataset they choose.

**Dataset keys to use in code**

* `breastmnist`
* `pneumoniamnist`

> (Do **not** pick non-binary sets like PathMNIST/DermaMNIST/BloodMNIST/ChestMNIST)

---

## 2) Repository Structure (suggested)

```
medmnist-ssl/
  ├─ notebooks/
  │   └─ ML_Basics_MedMNIST_Colab.ipynb   # provided starter
  ├─ starter/
  │   └─ ml_medmnist_starter_v2.zip       # optional CLI starter
  ├─ results/
  │   └─ week1/
  │       └─ <dataset_key>/
  │           └─ <your_name>/             # your personal folder
  │               ├─ test_metrics.json
  │               ├─ reliability.png
  │               ├─ screenshot_env.png
  │               └─ README_week1.md
  ├─ README.md
  └─ .gitignore
```

**.gitignore (minimum)**

```
runs/
__pycache__/
.ipynb_checkpoints/
*.pt
*.png
*.json
```

> We will copy **just the final artifacts** to `results/week1/...`. Do not commit large raw run folders.

---

## 3) Quickstart — Google Colab (recommended)

1. Open `notebooks/ML_Basics_MedMNIST_Colab.ipynb` in **Google Colab**.
2. Turn on GPU: **Runtime → Change runtime type → GPU**.
3. Run cells top→down. In the **config** cell, set:

   ```python
   DATASET_KEY = 'pneumoniamnist'  # or 'breastmnist' (pick ONE)
   MODEL_NAME  = 'resnet18'        # or 'smallcnn' if runtime is slow
   EPOCHS      = 5                 # Week 1 target
   BATCH_SIZE  = 128               # use 64/32 if you hit memory issues
   ```
4. The dataset will **auto-download** via the `medmnist` package and cache at `~/.medmnist` (Colab: `/root/.medmnist/`).
5. After training, verify `runs/` contains:

   * `best.pt`, `val_log.jsonl`, `test_metrics.json`, `reliability.png`
6. Copy the three artifacts + screenshot to your `results/week1/<dataset>/<your_name>/` folder.

**Optional: Save to Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
OUTDIR = '/content/drive/MyDrive/medmnist_runs'
# Use OUTDIR for saving models/figures if you want them persisted between sessions.
```

---

## 4) Alternative — Local (optional)

1. Unzip `starter/ml_medmnist_starter_v2.zip`.
2. Create a venv and install requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Run a 5-epoch baseline:

   ```bash
   # SmallCNN (faster)
   python -m src.train_medmnist --dataset pneumoniamnist --model smallcnn --epochs 5

   # ResNet18 (head-only finetune)
   python -m src.train_medmnist --dataset breastmnist --model resnet18 --epochs 5 --finetune head
   ```
4. Collect `test_metrics.json`, `reliability.png`, and a terminal screenshot.

---

## 5) Week 1 Tasks (what to submit)

Place the following under `results/week1/<dataset_key>/<your_name>/`:

* `test_metrics.json` — copy from your run (contains Accuracy/AUROC, etc.)
* `reliability.png` — reliability diagram (ECE on the title)
* `screenshot_env.png` — Colab or terminal screenshot
* `README_week1.md` (3–5 sentences):

  * Which dataset and model you used
  * Epochs + batch size (and `--finetune` if ResNet18)
  * One small issue you hit (if any) and how you fixed it
  * One observation about Acc/AUROC across epochs

**Rubric (Week 1, 10 pts for progress tracking)**

* (3) One successful 5-epoch run with artifacts present
* (3) `test_metrics.json` + `reliability.png` included
* (2) Clear short write-up
* (2) Repo hygiene (`results/` structure, `.gitignore` honored)

---

## 6) GitHub: simplest workflow

**Create the repo**

1. Go to GitHub → **New repository** → name: `medmnist-ssl` → Private → Create.
2. Click **Upload files** and add:

   * `notebooks/ML_Basics_MedMNIST_Colab.ipynb`
   * `starter/ml_medmnist_starter_v2.zip` (optional)
   * `results/week1/...` (after you run)

**Commit message examples**

* `Week1: add notebook + results (pneumoniamnist)`
* `Week1: baseline (breastmnist, smallcnn) + metrics`

> Tip: If files are large, only upload the three artifacts + screenshot.

---

## 7) Colab: common pitfalls

* **No GPU**: Runtime → Change runtime type → **GPU**, then rerun the install cell.
* **Slow runtime**: use `MODEL_NAME='smallcnn'`, reduce `BATCH_SIZE` to 64 or 32.
* **Import errors**: rerun the install cell (Colab resets packages when it restarts).
* **Permission/Path**: ensure the output directory exists before saving.

---

## 8) Research framing (why we do this)

* We’re preparing for **self-supervised contrastive learning (SSL)** in Weeks 4–6.
* Week 1 is about having a **working, reproducible baseline** for a **binary** medical task.
* Keep notes: we will compare **supervised vs. transfer vs. SSL** later on the same dataset.

---

## 9) Who does what (assignment)

* Each member **chooses one** binary dataset: `breastmnist` or `pneumoniamnist`.
* You will be **assigned** to that dataset for the project (keep using it in later weeks).
* Use your own subfolder: `results/week1/<dataset_key>/<your_name>/`.

---

## 10) FAQ

* **Can I change datasets later?** Prefer not to, for fair comparisons over time. If you must, note it clearly in your README.
* **Do I need exact numbers in Week 1?** No—just show that the pipeline runs and artifacts are produced. We’ll optimize later.
* **Where do we discuss issues?** Open a short GitHub Issue titled `Week1: <your_name> <dataset_key>` with your question and what you tried.

---

## 11) Next steps (preview)

* **Week 2–3**: EDA, SmallCNN vs ResNet18 (head/all), light augmentation sanity checks.
* **Weeks 4–6**: Required **SSL** (pretrain → linear probe → fine-tune) on your chosen dataset.
* **Weeks 7–10**: Reliability (ECE), error analysis, writing + short talk.
