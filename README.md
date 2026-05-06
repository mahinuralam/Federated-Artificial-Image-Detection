# FMACNN: Federated Multi-Attention CNN for AI-Generated Image Detection

Official implementation of the paper:

> **FMACNN: Federated Multi-Attention CNN Framework for Artificial Image Detection**
> Md Mahinur Alam, Mohtasin Golam, Taesoo Jun
> *Journal of Information Security and Applications*, vol. 100, p. 104462, 2026, Elsevier.

---

## Model Architecture

- Three convolutional blocks, each with double Conv2D layers, Batch Normalization, and L2 regularization (He normal initialisation)
- **Block 1** — Squeeze-and-Excitation (SE) channel attention + spatial attention
- **Block 2** — SE channel attention + dual-pooling channel attention (CBAM-style)
- **Block 3** — deep feature extraction without attention gating
- Fully connected head: Dense(512) → Dense(256) → Softmax, with Batch Normalization and Dropout at each layer

## Federated Learning Architecture

- **Protocol**: FedAvg with per-round client sampling and multiple local epochs
- **Client scaling**: weight contributions scaled by N_k / N (selected clients only, factors sum to 1)
- **Aggregation**: percentile-based weight clipping (95th) to suppress destabilising outlier updates
- **Stability monitoring**: L2 weight-divergence tracked across rounds
- **IID mode**: data randomly shuffled before sharding — uniform class distribution per client
- **Non-IID mode**: data sorted by class label before sharding — skewed, realistic heterogeneous distribution
- **Early stopping**: configurable accuracy target, wall-clock budget, and per-client batch cap

## Dataset — RealAIGI

- Realistic AI-Generated Image dataset published on IEEE DataPort
- **Download**: [https://dx.doi.org/10.21227/0da4-g645](https://dx.doi.org/10.21227/0da4-g645)
- Images normalised to `[0, 1]`, one-hot encoded labels, 90/10 train/test split
- Custom datasets are also supported: place class folders under `Datasets/<name>/<class>/` and point `configs/default.yaml` at the root

## Results

<img width="1053" height="576" alt="IID training curves" src="https://github.com/user-attachments/assets/b0ac5f78-b052-4b49-b737-d29d94d8390c" />
<img width="1198" height="900" alt="Non-IID training curves" src="https://github.com/user-attachments/assets/62407846-ff0e-4809-b1c3-bd966268d641" />
<img width="1314" height="330" alt="IID vs Non-IID comparison" src="https://github.com/user-attachments/assets/19600095-4159-4106-8802-ac500669a7a5" />

---

## Installation

```bash
git clone https://github.com/mahinuralam/Federated-Artificial-Image-Detection.git
cd Federated-Artificial-Image-Detection
pip install -e ".[dev]"
```

## Usage

```bash
# Full training (IID + Non-IID) with default config
python -m fmacnn --config configs/default.yaml

# Single mode
python -m fmacnn --mode iid
python -m fmacnn --mode noniid

# Quick smoke-test (5 rounds, reduced image size)
python -m fmacnn --config configs/debug.yaml

# Custom dataset path and output directory
python -m fmacnn --data-root /path/to/dataset --output-dir runs/exp1

# Run tests
pytest
```

All outputs (trained models, metrics JSON, plots) are written to `outputs/` by default.

---

## Citation

If you use this code or the RealAIGI dataset, please cite:

```bibtex
@article{alam2026fmacnn,
  title={FMACNN: Federated multi-attention CNN framework for artificial image detection},
  author={Alam, Md Mahinur and Golam, Mohtasin and Jun, Taesoo},
  journal={Journal of Information Security and Applications},
  volume={100},
  pages={104462},
  year={2026},
  publisher={Elsevier}
}

@data{0da4-g645-25,
  doi = {10.21227/0da4-g645},
  url = {https://dx.doi.org/10.21227/0da4-g645},
  author = {Md Mahinur Alam and Taesoo Jun},
  publisher = {IEEE Dataport},
  title = {RealAIGI: Realistic AI Generated Image Dataset},
  year = {2025}
}

@inproceedings{alam2025realaigi,
  title={RealAIGI: An Innovative Dataset for Enhanced Detection of Real and AI-Generated Images},
  author={Alam, Md Mahinur and Tanha, Kanita Jerin and Subhan, Md Raihan and Jun, Taesoo},
  booktitle={2025 International Conference on Mobile, Military, Maritime IT Convergence (ICMIC)},
  pages={145--146},
  year={2025},
  organization={IEEE}
}
```
