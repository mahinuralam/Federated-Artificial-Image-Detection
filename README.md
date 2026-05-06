# FMACNN: Federated Multi-Attention CNN for AI-Generated Image Detection

Official implementation of the paper:

> **FMACNN: Federated Multi-Attention CNN Framework for Artificial Image Detection**
> Md Mahinur Alam, Mohtasin Golam, Taesoo Jun
> *Journal of Information Security and Applications*, vol. 100, p. 104462, 2026, Elsevier.
> [Read the paper](https://www.sciencedirect.com/science/article/abs/pii/S221421262600092X)

---

## Overview

FMACNN is a privacy-preserving framework that detects AI-generated images without centralising data. Multiple clients train a shared model locally and only share model updates — never raw images.

## Model Architecture

- A CNN with built-in attention mechanisms that help the model focus on the most informative regions and channels of an image
- Three convolutional stages with progressively deeper feature extraction
- Designed to distinguish real photographs from AI-generated images

## Federated Learning

- Multiple clients train on their own local data and contribute to a shared global model
- Supports two data settings:
  - **IID** — each client has a balanced mix of classes (ideal scenario)
  - **Non-IID** — each client has a skewed class distribution (realistic scenario)
- Aggregation is done via weighted averaging with outlier suppression to keep training stable

## Dataset — RealAIGI

- A realistic dataset of real and AI-generated images
- Published on IEEE DataPort: [https://dx.doi.org/10.21227/0da4-g645](https://dx.doi.org/10.21227/0da4-g645)

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
# Run full training
python -m fmacnn --config configs/default.yaml

# Run IID or Non-IID only
python -m fmacnn --mode iid
python -m fmacnn --mode noniid

# Quick test run
python -m fmacnn --config configs/debug.yaml
```

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
