# Secure Federated Segmentation for Skin Cancer Diagnosis

A federated learning framework for dermoscopic image segmentation across five heterogeneous medical institutions. The system trains a single segmentation model without any institution sharing raw patient images. Each hospital trains locally and exchanges only model weights with a central server.

The proposed method, **DEP**, combines three ideas on top of standard federated averaging: elastic weight consolidation during local training, an exponential moving average on the server, and a per client personalisation step at the end of training.

This repository contains the full implementation, training scripts, configuration, and results that produced the numbers in our thesis report.

---

## Headline Results

Across five public dermoscopic datasets ranging from 185 to 6,904 training images per client:

| Stage | Mean Dice | Mean IoU | Mean Boundary IoU |
|:------|:---------:|:--------:|:-----------------:|
| Old baseline (scratch U-Net, BCE only)        | 0.872 | 0.797 | n/a   |
| New local pretrain (EffNet-B4 + augmentation) | 0.925 | 0.868 | 0.557 |
| DEP federated global model                    | 0.911 | 0.844 | 0.476 |
| **DEP + per-client personalisation**          | **0.938** | **0.888** | **0.589** |

Per client highlights after personalisation:

| Client | Dice | IoU | Note |
|:-------|:----:|:---:|:-----|
| C1 (HAM10000)  | 0.949 | 0.909 | Largest dataset, 6904 train images |
| C2 (ISIC-2018) | 0.922 | 0.862 | Medium dataset, 2531 train images |
| C3 (Med-Note)  | **0.964** | 0.932 | Smallest dataset, 185 train images. Crosses 0.95 Dice |
| C4 (SD-260)    | 0.910 | 0.840 | Minority client. Improved from 0.704 IoU in the old baseline |
| C5 (VIP-Skin)  | 0.945 | 0.899 | Improved from 0.743 IoU in the old baseline |

The cross client generalisation gap (own-data minus cross-client mean IoU) shrank from 0.212 in the old pipeline to 0.135 in this work.

---

## Why this Project Exists

Skin cancer diagnosis depends on accurate lesion segmentation. Deep learning has made automatic segmentation practical, but a model trained at one hospital often fails on images from another due to differences in imaging hardware, lighting, annotation style, and patient population. The standard fix, pooling data centrally, is not allowed under HIPAA, GDPR, and similar privacy laws.

Federated learning addresses this by keeping data local and only sharing model weights. Vanilla federated averaging works well when client datasets are similar, but real medical federations involve heterogeneous, non-IID partners. This project explores what combination of techniques produces a usable model under those realistic conditions.

---

## Repository Structure

```
SCD/
├── src/
│   ├── models.py              # smp factory + scratch U-Net
│   ├── datasets.py            # SkinCancerDataset + Albumentations
│   ├── losses.py              # BCE, Dice, Focal, combined
│   ├── metrics.py             # IoU, Dice, HD95, ASD, Boundary IoU
│   ├── utils.py               # config loader, logger, TTA, checkpoint
│   ├── evaluate.py            # full eval with TTA + boundary metrics
│   ├── train_local.py         # Phase 1: local pretraining
│   ├── train_fl.py            # Phase 2: federated training
│   ├── personalize.py         # Phase 3: per client fine-tune
│   ├── cross_client_eval.py   # 5x5 domain shift matrix
│   ├── analyze.py             # tables, figures, Wilcoxon tests
│   ├── compare_v3_v4.py       # old pipeline vs new pipeline
│   ├── fl/
│   │   ├── base.py            # FLMethod abstract base + FLContext
│   │   ├── fedavg.py          # plain federated averaging
│   │   ├── fedprox.py         # FedProx (Li et al. 2020)
│   │   ├── scaffold.py        # SCAFFOLD (Karimireddy et al. 2020)
│   │   ├── feddyn.py          # FedDyn (Acar et al. 2021)
│   │   ├── fedper.py          # FedPer (Arivazhagan et al. 2019)
│   │   ├── ditto.py           # Ditto (Li et al. 2021)
│   │   ├── moon.py            # MOON (Li et al. 2021)
│   │   └── dep.py             # DEP (this work)
│   └── dp/
│       ├── accounting.py      # privacy budget tracking via Opacus
│       └── __init__.py
├── configs/
│   └── base.yaml              # central configuration
├── experiments/
│   ├── run_phase2_dep.sh      # DEP fast-track pipeline
│   ├── run_all.sh             # full benchmark including DP sweep
│   ├── run_ablation.sh        # EWC on/off, EMA on/off
│   └── run_benchmark.sh       # all 8 FL methods
├── client_1/                  # HAM10000  (train, val, test folders)
├── client_2/                  # ISIC 2018
├── client_3/                  # Med-Note
├── client_4/                  # SD-260
├── client_5/                  # VIP-Skin
├── results/
│   ├── checkpoints/           # trained model .pth files
│   ├── logs/                  # per-run training logs
│   ├── tables/                # CSV + LaTeX result tables
│   └── figures/               # publication quality PNG figures
└── thesis_v2/                 # LaTeX thesis source + compiled PDF
```

---

## Installation

The project is built on PyTorch 2.9 with CUDA 12.8. It has been tested on a single NVIDIA RTX 5060 Ti (16 GB) under Ubuntu running inside WSL 2.

### Clone and Create Environment

```bash
git clone <repo-url> SCD
cd SCD
conda create -n scd python=3.10
conda activate scd
```

### Install Dependencies

```bash
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install segmentation-models-pytorch==0.5.0
pip install albumentations==2.0.8
pip install opacus==1.6.0
pip install monai==1.5.2
pip install timm pandas matplotlib pyyaml tqdm scipy scikit-learn
```

### Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Data Layout

Each client directory must follow this structure. Image and mask filenames must match exactly.

```
client_X/
├── train/
│   ├── images/   *.jpg, *.png, etc.
│   └── masks/    binary masks, same filenames as images
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Masks should be binary (lesion = 1, background = 0). The loader binarises at threshold 127 internally.

The five datasets used in this work are publicly available:

| Client | Source                          | Train | Val | Test |
|:-------|:--------------------------------|:-----:|:---:|:----:|
| C1     | HAM10000                        | 6904  | 1500| 1500 |
| C2     | ISIC 2018 Lesion Boundary Task  | 2531  | 554 | 558  |
| C3     | Med-Note                        | 185   | 8   | 10   |
| C4     | SD-260                          | 590   | 29  | 29   |
| C5     | VIP-Skin                        | 270   | 17  | 17   |

---

## How to Reproduce the Results

The full pipeline runs in three phases. Total wall clock time is around five hours on a single mid-range GPU.

### Phase 1: Local Pretraining

Train an EfficientNet-B4 U-Net at each client.

```bash
python -m src.train_local --clients 3,5,4,2,1
```

Outputs trained checkpoints to `results/checkpoints/v4_local_client{1..5}.pth` and a results table at `results/tables/v4_local_results.csv`.

Estimated time on a single RTX 5060 Ti: 1 hour 45 minutes.

### Phase 2: Federated Learning

Run DEP for ten rounds with five local epochs per round.

```bash
python -m src.train_fl --method dep --rounds 10 --local-epochs 5
```

Other available methods include `fedavg`, `fedprox`, `scaffold`, `feddyn`, `fedper`, `ditto`, `moon`. Select any of them by changing the `--method` flag.

Estimated time for ten rounds of DEP: 2 hours 30 minutes.

### Phase 3: Personalisation

After federated training finishes, fine-tune the global model per client.

```bash
python -m src.personalize \
    --from-checkpoint results/checkpoints/v4_dep_final.pth \
    --suffix dep
```

Outputs per-client personalised models to `results/checkpoints/v4_dep_personal_client{1..5}.pth`.

Estimated time: 25 minutes.

### Cross Client Evaluation

Generate the 5x5 domain shift matrix.

```bash
python -m src.cross_client_eval --tta
```

Outputs `results/tables/v4_cross_client_iou.csv` and `results/figures/FIG_cross_client_heatmap.png`.

### Tables and Figures

Generate all paper-ready tables and figures.

```bash
python -m src.analyze --all
python -m src.compare_v3_v4
```

### One-shot Pipeline

For convenience, the full Phase 2 sequence (analysis + cross-client + DEP + personalise + tables/figures) is wrapped in a single script:

```bash
bash experiments/run_phase2_dep.sh
```

The full benchmark across all eight federated methods and four differential privacy budgets is available at:

```bash
bash experiments/run_all.sh
```

This longer run takes approximately fifteen hours on a single GPU and is recommended for overnight execution.

---

## The DEP Method in Brief

DEP combines four ideas:

1. **FedAvg aggregation** (McMahan et al. 2017): the server averages client weights by dataset size.

2. **Elastic weight consolidation** (Kirkpatrick et al. 2017): during local training, each client adds a penalty that anchors important parameters near the global model. Importance is estimated by the diagonal of the Fisher information matrix computed once at the start of training.
   
   `L_total = L_task + (lambda / 2) * sum_i F_i * (theta_i - theta_global_i)^2`
   
   We use `lambda = 5000` and 150 samples for Fisher estimation.

3. **Server-side exponential moving average**: after each FedAvg step, the server updates a smoothed copy of the model with `w_ema = 0.9 * w_ema + 0.1 * w_fedavg`. The smoothed copy is what gets distributed in the next round. The EMA accumulator is bootstrapped from the first real round rather than from random initialisation.

4. **Per-client personalisation**: after the federated rounds finish, each client fine-tunes the global model on its own training data for five epochs at a lower learning rate. The personalised model is what each client deploys.

---

## Training Stack Choices

These choices matter at least as much as the federated learning algorithm. They are documented in detail in the thesis but summarised here.

| Component | Choice | Reason |
|:---|:---|:---|
| Encoder | EfficientNet-B4 ImageNet pretrained | Best accuracy/compute trade-off on our GPU |
| Decoder | Standard U-Net decoder via smp library | Well tested, stable |
| Input resolution | 384 x 384 | Higher resolution helps boundaries; 512 was tested but exceeded VRAM |
| Augmentation | Albumentations (flip, rotate, affine, elastic, color, dropout) | Essential for small clients |
| Loss | 0.3 BCE + 0.4 Dice + 0.3 Focal | Handles class imbalance, fast early convergence |
| Optimiser | AdamW with cosine schedule + warm up | Stable across clients |
| Mixed precision | enabled | Halves memory, modest speed gain |
| TTA | flip H + flip V + flip HV ensemble | Small but consistent gain at no extra training cost |
| Threshold | per client validation-tuned | Default 0.5 underperforms |

---

## What this Repository Does Not Yet Do

The codebase is functional but the empirical evaluation has some gaps that we are open about. These are listed here so users know what to expect.

- Only DEP was run for the full ten-round benchmark. The other seven federated methods are implemented and smoke-tested, but the head-to-head comparison across all methods is left as future work.
- All reported numbers come from a single random seed. Multi-seed reporting with statistical tests is straightforward to add and recommended for any publication-quality use.
- The differentially private (DP-SGD) code path is integrated and tested through Opacus, but the full privacy budget sweep (epsilon in {1, 3, 5, 10}) was not completed within the original time frame. The infrastructure is in place; only execution remains.
- External validation on a held-out dataset such as PH2 is not yet performed.
- No system design diagrams (block diagram, sequence diagram, data flow diagram) are included as image files. The descriptions in the thesis are prose-based.

These limitations are documented in the conclusion of the thesis and are not hidden.

---

## Configuration

All hyperparameters are controlled through `configs/base.yaml`. The most useful knobs are listed below.

```yaml
model:
  arch: "unet"
  encoder: "efficientnet-b4"
  encoder_weights: "imagenet"

img_size: 384

local:
  epochs: 50
  batch_size: 16
  lr: 3.0e-4
  patience: 10

fl:
  num_rounds: 10
  local_epochs: 5
  lr: 1.0e-4
  ewc_lambda: 5000
  ema_alpha: 0.9
  fisher_samples: 150

personalize:
  epochs: 5
  lr: 5.0e-5

dp:
  enabled: false
  target_epsilon: 5.0
  target_delta: 1.0e-5
  max_grad_norm: 1.0
```

Command line flags override the YAML defaults. For example:

```bash
python -m src.train_fl --method dep --rounds 20 --ewc-lambda 1000
```

---

## Output Files

After a full pipeline run, the following artefacts are produced.

### Checkpoints

- `results/checkpoints/v4_local_client{1..5}.pth` — local pretrained models
- `results/checkpoints/v4_dep_round{1..10}.pth` — per-round federated checkpoints
- `results/checkpoints/v4_dep_final.pth` — final federated global model
- `results/checkpoints/v4_dep_personal_client{1..5}.pth` — personalised models

### Tables

- `v4_local_results.csv` — intra-client baseline
- `v4_cross_client_iou.csv`, `v4_cross_client_dice.csv` — 5x5 matrices
- `v4_dep_history.csv` — round by round IoU
- `v4_dep_final_eval.csv` — final federated metrics
- `v4_dep_personal_final_eval.csv` — final personalised metrics
- `TABLE_DICE_HEADLINE.csv` — paper headline table
- `TABLE_three_stage_progression.csv` — Local vs DEP vs DEP+Personal
- `v3_vs_v4_local.csv` — old pipeline vs new pipeline

### Figures (220 DPI PNG)

- `FIG_v3_vs_v4_local.png` — pipeline upgrade impact
- `FIG_cross_client_heatmap.png` — 5x5 domain shift heatmap
- `FIG_convergence_per_client.png` — per-client convergence
- `FIG_three_stage_progression.png` — headline progression chart
- `FIG_method_bars_noDP.png` — method comparison bars

---

## Hardware Requirements

The full pipeline as configured here requires:

- One NVIDIA GPU with at least 8 GB of VRAM. 16 GB is comfortable.
- About 10 GB of disk space for checkpoints and result files.
- 16 GB of system RAM. Less may work but DataLoader workers will be tight.
- Approximately five hours of compute for a single full run including DEP and personalisation.

Lower-end hardware can run the pipeline with reduced batch size (set `batch_size: 8` in `configs/base.yaml`) and lower image resolution (set `img_size: 256`). Both changes will reduce the final metrics by a small amount.

---

## Citation

If you use this code in academic work, please cite the underlying thesis report. The references file `thesis_v2/references.bib` contains the standard citations for the methods used.

```
@thesis{scd2026federated,
  title  = {A Secure Federated Segmentation System for Skin Cancer Diagnosis
            Using U-Net and Distributed Medical Imaging Data},
  author = {Chakma, Tukhor and Shamayel, Abdullah Al and Rahman, Kholilur},
  school = {University of Asia Pacific},
  year   = {2026},
  type   = {Bachelor of Science Thesis}
}
```

---

## Acknowledgements

This work was supervised by Dr. Nasima Begum, Professor in the Department of Computer Science and Engineering at the University of Asia Pacific, and co-supervised by Rashik Rahman, Lecturer in the same department.

The implementation builds directly on a number of open source projects:

- [PyTorch](https://pytorch.org/) for the core deep learning framework
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) for pretrained U-Net backbones
- [Albumentations](https://github.com/albumentations-team/albumentations) for the augmentation pipeline
- [MONAI](https://monai.io/) for the boundary metrics
- [Opacus](https://opacus.ai/) for differentially private SGD support

The skin lesion datasets used in this study were collected and released by their respective owners (HAM10000, ISIC, and the named regional clinics) under terms permitting academic research use.

---

## License

This project is released under the MIT License. The datasets used remain under the licences of their original publishers.

---

## Contact

For questions about the code or the experimental setup, open an issue on this repository.

For questions about the underlying research, contact the supervising team at the University of Asia Pacific.
