# Self-Pruning Neural Network
### Tredence AI Engineering Intern — Case Study Submission

---

## What This Project Does

Implements a feed-forward neural network for **CIFAR-10 image classification** where
weights are pruned **during training** — not after it. Each weight has a learnable
**gate** (a scalar ∈ (0,1)); an L1 sparsity penalty drives unnecessary gates to zero,
effectively removing those weights from the network.

---

## Project Structure

```
self-pruning-neural-network/
│
├── prunable_network.py       ← ONLY file you need to run
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
│   (auto-generated after running the script)
├── gate_distribution.png     ← Gate histogram plot (deliverable)
├── report.md                 ← Markdown report with results (deliverable)
├── results.json              ← Raw λ / accuracy / sparsity numbers
└── data/                     ← CIFAR-10 downloaded here automatically
```

---

## ⚡ Quick Start (Step-by-Step)

### Step 1 — Prerequisites

Make sure you have **Python 3.9+** installed.  
Check: `python --version`

You also need **pip** (comes with Python).

---

### Step 2 — Clone / Download the project

If you received this as a ZIP:
```bash
unzip self-pruning-neural-network.zip
cd self-pruning-neural-network
```

If you cloned from GitHub:
```bash
git clone https://github.com/<your-username>/self-pruning-neural-network.git
cd self-pruning-neural-network
```

---

### Step 3 — Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1
```

You'll see `(venv)` appear in your terminal prompt when it's active.

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (CUDA):** If you have an NVIDIA GPU, replace the torch install
> with the CUDA version for ~5× faster training:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install numpy matplotlib
> ```

---

### Step 5 — Run the script

```bash
python prunable_network.py
```

**What happens automatically:**

| # | Action | Time estimate |
|---|--------|---------------|
| 1 | Downloads CIFAR-10 (~170 MB) into `./data/` | 1–3 min (first run only) |
| 2 | Trains model with **λ = 0.0001** for 40 epochs | ~10 min CPU / ~2 min GPU |
| 3 | Trains model with **λ = 0.001**  for 40 epochs | ~10 min CPU / ~2 min GPU |
| 4 | Trains model with **λ = 0.01**   for 40 epochs | ~10 min CPU / ~2 min GPU |
| 5 | Saves `gate_distribution.png` | instant |
| 6 | Saves `report.md` with actual result numbers | instant |
| 7 | Saves `results.json` | instant |

**Total time: ~30–40 min on CPU, ~6–8 min on GPU.**

---

### Step 6 — Check your deliverables

After the script finishes you will see:

```
─────────────────────────────────────────────────────
  ✅  All deliverables ready!
─────────────────────────────────────────────────────
  prunable_network.py   — source code (this file)
  gate_distribution.png — gate histogram plot
  report.md             — written Markdown report
  results.json          — raw λ / accuracy / sparsity data
─────────────────────────────────────────────────────
```

Open `report.md` in any Markdown viewer (VS Code, GitHub, Typora) to see the
final formatted report with the results table auto-filled.

---

## Expected Results

| λ (Lambda) | Approx Test Accuracy | Approx Sparsity |
|:----------:|:--------------------:|:---------------:|
| `1e-4` (low)    | ~50–54%        | ~10–25%         |
| `1e-3` (medium) | ~45–50%        | ~40–65%         |
| `1e-2` (high)   | ~35–45%        | ~70–90%         |

> Exact numbers depend on hardware, random seed, and CUDA version.
> Results are printed live and saved to `results.json` and `report.md`.

---

## Submitting to GitHub

```bash
git init
git add prunable_network.py requirements.txt README.md
git add gate_distribution.png report.md results.json   # after running
git commit -m "Tredence case study: Self-Pruning Neural Network"
git remote add origin https://github.com/<your-username>/self-pruning-neural-network.git
git push -u origin main
```

Share the GitHub repository URL in your submission.

---

## Key Code Concepts

### PrunableLinear Layer
```python
gates          = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
pruned_weights = self.weight * gates               # soft pruning
output         = F.linear(x, pruned_weights, self.bias)
```

### Total Loss
```
L_total = CrossEntropyLoss  +  λ × Σ sigmoid(gate_scores)
                                      all layers
```

### Sparsity Metric
```
Sparsity (%) = (# gates < 0.01) / (total gates)  × 100
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| Download fails / slow | CIFAR-10 needs internet. Use a hotspot or re-run. |
| `RuntimeError: CUDA out of memory` | Lower batch size: change `128` → `64` in `get_cifar10_loaders()` |
| Script is very slow | Add `--device cpu` note: the script auto-detects GPU. Ensure CUDA torch is installed for GPU. |
| `report.md` is missing | Script must **finish completely** — wait for `✅ All deliverables ready!` |

---

*Built for Tredence AI Engineering Intern Case Study — 2025 Cohort*
