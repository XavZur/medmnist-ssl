# medmnist-ssl — Week 1 README (Kickoff & Environment Setup)

**Audience:** U0–U1 (McGill COMP 202/250/206 level)
**Goal (Week 1):** Run one end-to-end training on MedMNIST and push minimal artifacts to **GitHub**.

> This repo hosts the Week 1 materials for the **medmnist-ssl** research project (self-supervised contrastive learning with MedMNIST). Week 1 focuses on getting the pipeline running on **Google Colab** and organizing outputs in GitHub.

---

## 🔖 Repo Description (use in GitHub “Description”)

Self-supervised contrastive learning on MedMNIST: pretrain → linear probe → fine-tune, with label-efficiency and calibration. (Week 1: setup & baseline run.)

**Suggested topics:** `medmnist, self-supervised-learning, contrastive-learning, simclr, moco, pytorch, medical-imaging, label-efficiency, calibration, education, undergraduate, colab, reproducibility`

---

## 🗂️ Suggested Repository Structure

```
medmnist-ssl/
  ├─ notebooks/
  │   └─ ML_Basics_MedMNIST_Colab.ipynb
  ├─ starter/
  │   └─ ml_medmnist_starter_v2.zip
  ├─ results/
  │   └─ week1/              # your outputs in Week 1
  ├─ figures/                # (optional) plots you keep
  ├─ reports/                # (optional) short write-ups later
  ├─ README.md               # this file
  └─ .gitignore
```

**.gitignore (Week 1 minimal):**

```
runs/
__pycache__/
.ipynb_checkpoints/
*.pt
*.png
*.json
```

> Rationale: keep large/binary outputs out of git history. Copy only the few required artifacts to `results/week1/` for submission.

---

## 🚀 Quick Start (for Week 1)

1. **Create a GitHub repo** named **`medmnist-ssl`** (Private recommended).
2. **Upload** two starter files:

   * `notebooks/ML_Basics_MedMNIST_Colab.ipynb`
   * `starter/ml_medmnist_starter_v2.zip` (optional for local)
3. **Open the notebook in Colab** and enable **GPU**, then run cells top → bottom.
4. **Verify outputs** were created (`runs/.../test_metrics.json`, `reliability.png`).
5. **Copy required artifacts** to `results/week1/` and **commit** to GitHub.

---

## 🧑‍💻 Using GitHub (Beginner-friendly)

### Option A — Easiest (Web Only)

1. Go to GitHub → **New repository** → name it `medmnist-ssl` → **Private** → Create.
2. Click **Add file → Upload files**.
3. Drag-drop the notebook and (optionally) the starter zip into the right folders (`notebooks/`, `starter/`).
4. Click **Commit changes**.

### Option B — GitHub Desktop (No terminal needed)

1. Install **GitHub Desktop** → **File → New repository** → `medmnist-ssl`.
2. Add the files locally (place notebook under `notebooks/` etc.).
3. **Commit to main** → **Publish repository** (Private).

### Option C — Command Line (optional)

```bash
git clone https://github.com/<yourname>/medmnist-ssl.git
cd medmnist-ssl
git add .
git commit -m "Week1: baseline run + results"
git push origin main
```

> Keep commit messages short and meaningful, e.g., `Week1: setup colab + 5-epoch run`.

---

## 🧪 Google Colab: How to Run

### Open the notebook

* In GitHub, open `notebooks/ML_Basics_MedMNIST_Colab.ipynb` → use the **Open in Colab** button (if present) or download and upload to Colab.

### Turn on GPU

* Colab menu: **Runtime → Change runtime type → Hardware accelerator → GPU** → Save.

### Run cells

* Run the **install** cell (installs `torch`, `torchvision`, `medmnist`, etc.).
* In the **config** cell, for Week 1 use:

```python
DATASET_KEY = 'pathmnist'   # start here
MODEL_NAME  = 'resnet18'    # or 'smallcnn' if slow
EPOCHS      = 5             # Week 1 target
BATCH_SIZE  = 128           # reduce to 64/32 if needed
```

* Run the **training** cells. First run will auto-download data to `~/.medmnist` (Colab: `/root/.medmnist/`).

### (Optional) Save outputs to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
OUTDIR = '/content/drive/MyDrive/medmnist_runs'  # use as your output path
```

---

## 📦 Expected Outputs (Week 1)

After training (5 epochs), check **`runs/<dataset>_<model>_frac1.0_seed42/`**:

* `best.pt` — best model checkpoint (don’t commit to git)
* `val_log.jsonl` — per-epoch val metrics
* `test_metrics.json` — final Acc/AUROC (copy this one)
* `reliability.png` — calibration plot (copy this one)

> For GitHub submission, copy only the required artifacts to `results/week1/`.

---

## 📥 What to Commit for Week 1

Place these under **`results/week1/`** and push to GitHub:

* `screenshot_env.png` — Colab or terminal screenshot
* `test_metrics.json` — from your run’s `runs/.../`
* `reliability.png` — from your run’s `runs/.../`
* `README_week1.md` — 3–5 sentences: model, epochs, batch size, one issue you solved

**Rubric (10 pts for progress tracking)**

* (3) One successful 5-epoch run with outputs
* (3) `test_metrics.json` + `reliability.png` present
* (2) Short write-up (clarity)
* (2) Repo hygiene (structure + .gitignore)

---

## 🧰 Troubleshooting (Fast)

* **No GPU** → Runtime → *Change runtime type* → GPU, then re-run install cell.
* **Too slow** → set `MODEL_NAME='smallcnn'`, lower `BATCH_SIZE` to 64 or 32.
* **Import errors** → re-run the install cell; for local, `pip install -r requirements.txt`.
* **Permission/path errors** → ensure output directory exists; use `OUTDIR` on Drive.
* **Runtime reset** → always re-run the install cell first.

---

## ❓ FAQ

**Q1. Do I need the starter zip for Week 1?**
A. Not strictly. The Colab notebook is enough. The zip is handy for local runs.

**Q2. Where does MedMNIST download to?**
A. `~/.medmnist/` (Colab: `/root/.medmnist/`). It downloads automatically on first use.

**Q3. Which dataset/model should I use first?**
A. `pathmnist` + `resnet18` (or `smallcnn` if slow), `EPOCHS=5`.

**Q4. What if validation accuracy doesn’t improve?**
A. It’s fine for Week 1. We’ll analyze next week. Check that GPU is enabled and batch size is reasonable.

**Q5. What should NOT be committed?**
A. Large artifacts (full `runs/`), model weights (`*.pt`), raw caches. Commit only the Week 1 subset in `results/week1/`.

---

## 🧭 Next Weeks (preview)

* **W2:** EDA, misclassification gallery, baselines (SmallCNN vs ResNet18).
* **W3:** Fine-tune-all and augmentation sanity checks.
* **W4–6:** **Required SSL** — pretrain (SimCLR/MoCo-lite) → linear probe → fine-tune.
* **W7–10:** Calibration/robustness, writing, figures, short talk.

---

## 📄 License & Acknowledgments (placeholders)

* Code snippets derived from educational examples; adapt under your course/research policy.
* MedMNIST dataset: please cite the original authors (see their documentation).

---

### ✅ Week 1 Definition of Done (DoD)

* [ ] Notebook runs end-to-end for 5 epochs
* [ ] `test_metrics.json` + `reliability.png` produced
* [ ] Artifacts pushed to `results/week1/` in **medmnist-ssl**
* [ ] Short write-up committed
