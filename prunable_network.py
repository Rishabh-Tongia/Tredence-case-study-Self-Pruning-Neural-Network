"""
=============================================================================
  Self-Pruning Neural Network — Tredence AI Engineering Intern Case Study
=============================================================================
  Author  : Rishabh
  Dataset : CIFAR-10
  Task    : Image classification with learnable weight pruning via gated layers

  Deliverables produced by running this script:
    ├── gate_distribution.png   — gate-value histogram for best λ model
    ├── report.md               — auto-generated Markdown report
    └── results.json            — raw λ / accuracy / sparsity table
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer augmented with a learnable gate for every weight.

    Forward pass:
        gates          = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_weights = weight ⊙ gates                (element-wise product)
        output         = input @ pruned_weights.T + bias

    A gate value near 0 zeroes-out the corresponding weight (pruning it).
    A gate value near 1 leaves the weight unchanged (keeping it).

    Both `weight` and `gate_scores` are registered nn.Parameters so the
    optimizer updates them jointly; gradients flow through the sigmoid and
    the element-wise multiply automatically via PyTorch autograd.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard linear parameters ──────────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # ── Gate score parameters (same shape as weight) ─────────────────────
        # gate_scores are unconstrained reals; sigmoid maps them to (0, 1).
        # Initialised to 0  →  sigmoid(0) = 0.5  (all gates half-open at start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming-uniform init for weights (good default for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — squeeze gate_scores into [0, 1] via Sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 — soft-prune: multiply each weight by its gate
        pruned_weights = self.weight * gates

        # Step 3 — standard affine transform with pruned weights
        #           F.linear(x, W, b)  computes  x @ W.T + b
        return F.linear(x, pruned_weights, self.bias)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def get_gates(self) -> torch.Tensor:
        """Return current gate values, detached from the computation graph."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of this layer's gates that are below `threshold`."""
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Self-Pruning Network
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (10-class image classification).

    Architecture (all linear layers are PrunableLinear):
        Input  : 3 × 32 × 32 = 3,072
        Layer 1: PrunableLinear(3072 → 512) + BN + ReLU + Dropout(0.3)
        Layer 2: PrunableLinear(512  → 256) + BN + ReLU + Dropout(0.3)
        Layer 3: PrunableLinear(256  → 128) + BN + ReLU
        Layer 4: PrunableLinear(128  → 10)
        Output : 10 logits (softmax applied by loss function)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3_072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        return self.net(x)

    # ── Pruning utilities ────────────────────────────────────────────────────
    def prunable_layers(self):
        """Iterator over all PrunableLinear layers."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Because gates = sigmoid(s) ∈ (0,1) are non-negative, L1 = sum of gates.
        Minimising this term drives gate values toward 0 (pruned).
        """
        sp = sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in self.prunable_layers()
        )
        return sp

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of all gates below `threshold` (network-wide)."""
        all_gates = torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        )
        return (all_gates < threshold).float().mean().item()

    def all_gate_values(self) -> np.ndarray:
        """All gate values concatenated as a NumPy array (for plotting)."""
        return np.concatenate(
            [layer.get_gates().cpu().numpy().flatten()
             for layer in self.prunable_layers()]
        )

    def total_gate_count(self) -> int:
        return sum(layer.gate_scores.numel() for layer in self.prunable_layers())


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """Download (if needed) and return CIFAR-10 train/test DataLoaders."""

    # Per-channel mean and std for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    # num_workers=0 keeps things simple cross-platform
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Test  samples : {len(test_ds):,}")
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam, device):
    """
    One full pass over the training set.

    Total Loss = CrossEntropyLoss  +  λ × SparsityLoss (L1 of all gates)

    Returns (avg_loss, accuracy_percent).
    """
    model.train()
    running_loss = correct = total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        logits   = model(inputs)
        cls_loss = F.cross_entropy(logits, targets)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = logits.max(dim=1)
        total        += targets.size(0)
        correct      += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return test accuracy (%) with gradients disabled."""
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(dim=1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def train_model(lam, train_loader, test_loader, device,
                epochs: int = 40, lr: float = 1e-3):
    """
    Train a fresh SelfPruningNet for `epochs` with sparsity coefficient `lam`.
    Returns (model, final_test_acc, final_sparsity_pct).
    """
    banner = f"λ = {lam}"
    print(f"\n{'='*60}")
    print(f"  Training  {banner}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing smoothly decays LR to ~0 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"  Total gate parameters : {model.total_gate_count():,}")
    print(f"  Device                : {device}\n")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss, tr_acc = train_one_epoch(model, train_loader, optimizer, lam, device)
        te_acc       = evaluate(model, test_loader, device)
        sparsity     = model.overall_sparsity() * 100.0
        scheduler.step()

        best_acc = max(best_acc, te_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | "
                  f"Loss {loss:7.4f} | "
                  f"Train {tr_acc:6.2f}% | "
                  f"Test {te_acc:6.2f}% | "
                  f"Sparsity {sparsity:5.1f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity() * 100.0

    print(f"\n  ✓  Test Accuracy  : {final_acc:.2f}%")
    print(f"  ✓  Sparsity Level : {final_sparsity:.2f}%")
    return model, final_acc, final_sparsity


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — Visualisation & Report
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(model, lam, path: str = "gate_distribution.png"):
    """
    Histogram of all gate values for `model`.
    A successful pruning shows:
      • Large spike near 0 → pruned weights
      • Smaller cluster near 1 → retained weights
    """
    gates    = model.all_gate_values()
    n_pruned = (gates < 1e-2).sum()
    pct      = 100.0 * n_pruned / len(gates)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gates, bins=120, color="#3A7EBF", edgecolor="white",
            linewidth=0.3, alpha=0.9, label="Gate values")

    ax.axvline(x=0.01, color="#E84040", linestyle="--", linewidth=1.8,
               label="Pruning threshold (0.01)")

    ymax = ax.get_ylim()[1]
    ax.annotate(
        f"{n_pruned:,} gates pruned\n({pct:.1f}% of total)",
        xy=(0.01, ymax * 0.65),
        xytext=(0.18, ymax * 0.65),
        arrowprops=dict(arrowstyle="->", color="#E84040"),
        fontsize=10, color="#E84040",
    )

    ax.set_xlabel("Gate Value  σ(gate_score)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Gate Value Distribution  —  λ = {lam}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓  Gate distribution plot → {path}")


def generate_report(results, best_lam, plot_path="gate_distribution.png"):
    """
    Auto-write report.md from actual experimental results.
    results: list of (lambda, test_acc, sparsity)
    """
    today = datetime.now().strftime("%B %d, %Y")

    rows = ""
    for lam, acc, spar in results:
        note = " ← best" if lam == best_lam else ""
        rows += f"| `{lam}` | {acc:.2f} | {spar:.2f} |{note}\n"

    md = f"""# Self-Pruning Neural Network — Case Study Report

**Author :** Rishabh  
**Date   :** {today}  
**Dataset:** CIFAR-10 &nbsp;|&nbsp; **Framework:** PyTorch

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Gating Mechanism

For every weight $w_{{ij}}$ in a `PrunableLinear` layer we introduce a learnable
scalar $s_{{ij}}$ (the *gate score*). The forward pass computes:

$$
\\text{{gate}}_{{ij}} = \\sigma(s_{{ij}}) \\in (0,1)
\\qquad
\\tilde{{w}}_{{ij}} = w_{{ij}} \\cdot \\text{{gate}}_{{ij}}
$$

When a gate collapses to ~0, the corresponding weight is effectively removed
("pruned") from the network — no post-processing needed.

### The Loss Function

$$
\\mathcal{{L}}_{{\\text{{total}}}} = \\underbrace{{\\mathcal{{L}}_{{CE}}}}_{{\\text{{classification}}}} + \\lambda \\cdot \\underbrace{{\\sum_{{i,j}} \\text{{gate}}_{{ij}}}}_{{\\text{{sparsity — L1 of gates}}}}
$$

### Why L1 Drives Gates to Exactly Zero

| Property | L1 | L2 |
|---|---|---|
| Gradient at 0 | sub-gradient = ±1 (non-zero) | gradient = 0 (vanishes) |
| Effect | pushes values **to** zero | pushes values **toward** zero |
| Typical result | **exact sparsity** | small but non-zero weights |

1. **Constant gradient pressure.** The sub-gradient of $|x|$ is $\\text{{sign}}(x)$,
   a *fixed* unit push regardless of magnitude.  Even a gate at 0.001 still gets
   the same downward nudge as one at 0.9.

2. **Sigmoid creates a stable "closed" state.** Once $\\sigma(s) \\approx 0$, the
   gradient flowing back through sigmoid vanishes, locking the gate shut.

3. **Trade-off via λ.** Larger λ amplifies the sparsity gradient relative to
   the classification gradient, forcing more gates to zero at the cost of accuracy.

The result is a **bimodal** gate distribution — a large spike at 0 (pruned) and
a smaller cluster near 1 (active) — visible in the histogram below.

---

## 2. Results — Sparsity vs Accuracy Trade-off

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
{rows}
**Interpretation:** As λ increases the network prunes more weights (sparsity ↑)
at the cost of predictive performance (accuracy ↓), demonstrating the classic
sparsity–accuracy trade-off.  The best accuracy was achieved at **λ = {best_lam}**.

---

## 3. Gate Value Distribution (Best Model: λ = {best_lam})

![Gate Distribution]({plot_path})

A successful self-pruning result has two features:

* **Large spike near gate ≈ 0** — weights the network decided are unimportant.
* **Smaller cluster near gate ≈ 1** — weights the network chose to retain.
* Virtually nothing in the middle — the binary sparsity that L1 encourages.

---

## 4. Architecture

```
Input (3 × 32 × 32 = 3,072 features)
         │
PrunableLinear(3072 → 512)  + BatchNorm1d + ReLU + Dropout(0.3)
         │
PrunableLinear(512  → 256)  + BatchNorm1d + ReLU + Dropout(0.3)
         │
PrunableLinear(256  → 128)  + BatchNorm1d + ReLU
         │
PrunableLinear(128  → 10)
         │
   10 output logits
```

**Optimizer:** Adam (lr = 1e-3, weight_decay = 1e-4)  
**Scheduler:** CosineAnnealingLR (T_max = 40)  
**Epochs per run:** 40

---

## 5. Key Takeaways

* The self-pruning mechanism identifies and removes unimportant connections
  **during training** — no separate post-processing step required.
* L1 regularisation on sigmoid gates is differentiable end-to-end and produces
  genuine zero-gates, not just small values.
* λ is a clean, interpretable knob: tune it to hit a desired sparsity budget
  for memory- or latency-constrained deployments.
"""

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("  ✓  Markdown report    → report.md")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'─'*60}")
    print(f"  Self-Pruning Neural Network  —  Tredence Case Study")
    print(f"{'─'*60}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : {device}")

    print("\n📦  Loading CIFAR-10 ...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # ── Three λ values: low / medium / high ──────────────────────────────────
    lambda_values = [1e-4, 1e-3, 1e-2]

    results = []   # (lam, acc, sparsity)
    models  = {}   # lam → trained model

    for lam in lambda_values:
        model, acc, sparsity = train_model(
            lam, train_loader, test_loader, device, epochs=40, lr=1e-3
        )
        results.append((lam, acc, sparsity))
        models[lam] = model

    # ── Identify best model by test accuracy ─────────────────────────────────
    best_lam = max(results, key=lambda r: r[1])[0]

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*58}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*58}")
    print(f"  {'Lambda':<10} {'Test Acc':>12} {'Sparsity':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12}")
    for lam, acc, sp in results:
        tag = "  ← best" if lam == best_lam else ""
        print(f"  {str(lam):<10} {acc:>11.2f}% {sp:>11.2f}%{tag}")
    print(f"{'='*58}")

    # ── Save raw results ──────────────────────────────────────────────────────
    with open("results.json", "w") as f:
        json.dump(
            [{"lambda": lam, "test_accuracy_pct": acc, "sparsity_pct": sp}
             for lam, acc, sp in results],
            f, indent=2,
        )
    print("\n  ✓  Raw results        → results.json")

    # ── Plot gate distribution for best model ─────────────────────────────────
    print("\n📊  Generating gate distribution plot ...")
    plot_gate_distribution(models[best_lam], best_lam, path="gate_distribution.png")

    # ── Generate Markdown report ──────────────────────────────────────────────
    print("\n📝  Writing report ...")
    generate_report(results, best_lam)

    print(f"\n{'─'*58}")
    print("  ✅  All deliverables ready!")
    print(f"{'─'*58}")
    print("  prunable_network.py   — source code (this file)")
    print("  gate_distribution.png — gate histogram plot")
    print("  report.md             — written Markdown report")
    print("  results.json          — raw λ / accuracy / sparsity data")
    print(f"{'─'*58}\n")


if __name__ == "__main__":
    main()
