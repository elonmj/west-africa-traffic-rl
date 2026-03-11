# Multi-Class ARZ + RL Traffic Signal Control (West Africa)

A reproducible Python implementation of a **multi-class macroscopic traffic model** (motorcycles + cars) coupled with **Deep Reinforcement Learning (DQN)** for adaptive signal control on a real 2×2 grid network extracted from OpenStreetMap.

This repository is the codebase associated with the paper:

> **A Multi-Class Macroscopic Traffic Model with Reinforcement Learning Signal Control for Heterogeneous Urban Networks in West Africa**
>
> Submitted to the *International Journal of Intelligent Transportation Systems Research* (Springer Nature).

## Why this project matters

Most RL traffic-signal papers assume data-rich environments (detectors, calibrated microsimulators, expensive tooling). This implementation demonstrates a practical pathway for **data-scarce cities**:

- model heterogeneous traffic (motos + cars) with a multi-class ARZ formulation,
- extract real urban topology from OpenStreetMap (Quartier Ganhi, Cotonou, Benin),
- train an adaptive controller on a standard CPU,
- evaluate across multiple demand regimes,
- generate publication-ready figures from the same pipeline.

## Key results

**Network case study** (2×2 grid, Quartier Ganhi, Cotonou):

| Scenario   | Improvement vs fixed timing |
|------------|----------------------------|
| Light      | **+83.8%**                 |
| Moderate   | **+55.7%**                 |
| Heavy      | **+34.8%**                 |
| Saturated  | **+29.0%**                 |
| **Overall**| **+45.7%**                 |

- Training budget: **60,000 steps** (~53 min on a standard CPU)
- Hardware: standard CPU (no GPU required)

## Repository structure

```text
├─ params.py                  # Physical parameters, scenarios, DQN config
├─ solver.py                  # Multi-class ARZ finite-volume solver (LxF)
├─ environment.py             # Single-link Gymnasium environment + baseline
├─ train.py                   # Single-link training pipeline
├─ network_params.py          # 2×2 grid network parameters (OSM-extracted)
├─ network_solver.py          # Network-level multi-link solver
├─ network_env.py             # Network Gymnasium environment + baseline
├─ network_train.py           # Network training/evaluation pipeline (paper)
├─ validate_riemann.py        # Riemann solver convergence validation
├─ fetch_real_topology.py     # OSM topology extraction script
├─ generate_all_figures.py    # Single-link figure generation
├─ generate_network_figures.py# Network figure generation (paper figures)
└─ data/results/              # Saved metrics, eval outputs, model artifacts
```

## Installation

### 1) Clone and enter folder

```bash
git clone https://github.com/elonmj/west-africa-traffic-rl.git
cd west-africa-traffic-rl
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

- **Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Full pipeline — Network (paper results)

```bash
python network_train.py
```

Runs:
1. environment sanity check,
2. proof phase (10k steps),
3. full training phase (60k steps),
4. multi-scenario evaluation on the 2×2 grid,
5. metrics export in `data/results/`.

### Optional modes

```bash
python network_train.py --proof-only
python network_train.py --full-only
```

### Single-link pipeline (development baseline)

```bash
python train.py
```

### Generate figures

```bash
python generate_network_figures.py   # Paper figures (network)
python generate_all_figures.py       # Single-link figures
```

## Reproducibility notes

- **Network topology:** 2×2 grid extracted from OpenStreetMap (Quartier Ganhi, Cotonou, Benin)
- **Road segments:** Avenue du Capitaine Adjovi, Avenue Augustin Nikouè, Rue Général Félix Éboué, Rue José Firmin Santos
- **Total network length:** 1,159 m
- **Action space (network):** `Discrete(16)` — 4 binary signal phases (one per junction)
- **Observation space (network):** 24-dimensional normalized state
- **Numerical scheme:** Local Lax-Friedrichs + forward Euler + CFL 0.5
- **Scenarios:** `light`, `moderate`, `heavy`, `saturated`
<!-- 
## Cite this work

If you use this code, please cite the associated paper (add DOI/link once public):

```bibtex
@article{hontinfinde_ahouanye_2026,
  title   = {A Multi-Class Macroscopic Traffic Model with Reinforcement Learning Signal Control for Heterogeneous Urban Networks in West Africa},
  author  = {Hontinfinde, Régis Donald and Ahouanye, Elonm Josaphat},
  journal = {International Journal of Intelligent Transportation Systems Research},
  year    = {2026}
}
``` -->

## Contact

- Régis Donald Hontinfinde — donald.hontinfinde@yahoo.com
- Elonm Josaphat Ahouanye — josaphatahouanye@gmail.com
