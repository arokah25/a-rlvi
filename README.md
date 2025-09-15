# A-RLVI: Amortized Robust Learning via Variational Inference

A-RLVI is a deep-learning extension of RLVI (Robust Learning via Variational Inference) in which a compact **inference network** predicts a per-sample *cleanliness* belief $\pi_i \in (0,1)$ from backbone features. These beliefs are used to (i) **robustify** supervised training and (ii) provide an **interpretable outlier score** at inference time.

- **Primary dataset:** Food-101 (real label noise)  
- **Backbone:** ImageNet-pretrained ResNet (default ResNet-50)  
- **Report:** [A_RLVI_report.pdf](https://github.com/user-attachments/files/22330674/A_RLVI_report.pdf) (derivations, design choices, ablations, results)

---

## Key idea (teacher-only, collapse-avoiding A-RLVI)

Classical RLVI updates per-sample corruption variables via a fixed-point rule driven mainly by loss values; this is not end-to-end differentiable and ignores richer feature signals. A-RLVI replaces that with a *shared*, learnable function $\pi_i=\sigma(f_\phi(z_i))$, enabling amortized, differentiable inference tied to the representation.

To avoid collapse feedback (shrinking $\pi$ to reduce the weighted CE), this implementation:
- **Detaches** $\pi$ from the classifier loss and **mean-normalizes** the weights so $\nabla_\phi L_\theta = 0$.
- Trains the inference net **only** against a *detached teacher* built from **batch z-scored** CE:  
  `r_i = zscore(CE_i)`,  
  `q_i(tau) = sigma(- r_i / tau + beta)`, with `beta = log( pi_bar / (1 - pi_bar) )` where `pi_bar` is an EMA of mean $\pi$.
- Minimizes:  
  `(1/B) * sum_i KL( Bern(pi_i) || Bern(q_i) )`  (teacher-only objective for the inference net).

Result: stable joint training, robust weighting, and informative $\pi$ distributions for auditing and outlier discovery.

---

## What this repository provides

- **A-RLVI (z-score teacher variant)** with:
  - Detached $\pi$ weighting for the classifier (prevents collapse loops).
  - Batch z-scored teachers and an EMA prior calibration.
  - OneCycleLR scheduling per parameter group (backbone/head/inference).
- **RLVI baseline** for comparison (deterministic E-step view on train).
- **Diagnostics and artifacts** saved per run:
  - CE/KL curves, grad norms, LR traces.
  - $\pi$ histogram and $\pi \rightarrow$ correctness by bins.
  - Test-accuracy curves and overlays across runs.
  - **Outlier export:** annotated grid of lowest-$\pi$ test samples + CSV metadata.

---

## Repository structure (essentials)

```
deep-learning/
  main.py                    # Runner: data, models, schedulers, early stopping, logging, outlier export
  train_arlvi_zscore.py      # A-RLVI (teacher-only z-score) single-epoch trainer
  train_rlvi.py              # implementation of the baseline RLVI trainer
  methods.py                 # Method routing (regular, rlvi, arlvi_zscore, arlvi_bayes, etc.)
  amortized/
    inference_net.py         # InferenceNet (LayerNorm + MLP → sigmoid pi)
  models/
    resnet.py, lenet.py      # Backbones (ResNet50 default for Food-101)
  data_load.py               # Food-101 loading & stratified train/val/test splits
  utils.py, data_tools.py    # Evaluation, printing, helpers
A_RLVI_report.pdf            # Project report (math, ablations, results)
```

---

## Installation

- Python 3.10+, PyTorch 2.x, torchvision 0.15+ recommended.
- Install dependencies:

```
pip install -r requirements.txt
```

A GPU is recommended; the runner auto-selects `cuda` / `mps` / `cpu`.

---

## Usage overview (minimal)

- Entry point: `deep-learning/main.py`
- Choose method via `--method`:  
  `arlvi_zscore` (recommended), `arlvi_bayes` (experimental), `rlvi`, plus standard baselines (`regular`, `coteaching`, `jocor`, `cdr`, `usdnl`, `bare`).
- Typical knobs:
  - A-RLVI: `--tau` (teacher temperature), `--ema_alpha` (EMA momentum), `--lr_inference`, `--wd_inference`, `--warmup_epochs`.
  - Training: `--batch_size`, `--n_epoch`, `--seed`, `--early_stop`, `--early_stop_patience`, `--eval_val_every`, `--eval_test_every`.
  - Data & paths: `--dataset food101`, `--root_dir`, `--result_dir`, `--download/--no-download`.

**Outputs (for `dataset=food101`, `method=arlvi_zscore`):**
```
<result_dir>/food101/arlvi_zscore/
  best_s<seed>.pt
  plots/
    losses_ce_kl.png
    grad_norms.png
    pi_histogram_arlvi_zscore.png
    lr_traces.png
    pi_to_correctness.png
    test_accuracy_over_epochs.png
  histories/
    test_acc_<run_label>_<timestamp>.npz
  outliers/
    top_outliers_grid_k10.png
    top_outliers_k10.csv
```

---

## Results (summary)

- **Accuracy parity with RLVI** on Food-101 under comparable schedules.
- **Belief quality:** higher $\pi$ bins consistently show higher accuracy; lowest-$\pi$ samples are concentrated in the exported outlier grid and are visually inspectable.
- See `A_RLVI_report.pdf` for derivations, ablations (teacher temperature, EMA momentum), and extended plots.

---

## Citing

Please cite RLVI and this implementation/report.

**RLVI:**
```
@inproceedings{karakulev2024adaptive,
  title={Adaptive robust learning using latent Bernoulli variables},
  author={Karakulev, Aleksandr and Zachariah, Dave and Singh, Prashant},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={23105--23122},
  year={2024}
}
```

**This repository/report (placeholder):**
```
@misc{rokah2025arlvi,
  title={Amortized Robust Learning via Variational Inference (A-RLVI)},
  author={Rokah, Adam},
  year={2025},
  note={Project report and code},
  howpublished={\url{https://github.com/<your-username>/a-rlvi}}
}
```

---

## License

Specify your license (e.g., MIT). Ensure compliance with external model/dataset licenses.

---

## Acknowledgments

RLVI conceptual foundation by A. Karakulev, D. Zachariah, and P. Singh (ICML 2024).  
This repository implements an amortized, deep-learning–friendly extension with diagnostics and outlier export for Food-101.
