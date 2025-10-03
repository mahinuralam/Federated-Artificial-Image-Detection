# Federated Multi-Attention CNN (FMACNN)

Federated pipeline for detecting AI-generated imagery with an attention-augmented CNN backbone.

## Overview

- Federated training of a Multi-Attention CNN (MACNN) for detecting AI-generated imagery.
- MacNN blocks: SE channel attention, spatial attention, L2-regularized conv stacks, fully connected head.
- Federated loop: client sampling, multiple local epochs, weighted FedAvg, percentile clipping, weight-divergence tracking, IID and Non-IID modes.

## Data

1. Default pipeline: CIFAR-10 loader normalizes to `[0,1]`, one-hot encodes labels, then splits 90/10.
2. Custom dataset: place class folders under `Datasets/<name>/class_x/*.jpg`; use the notebook cell that wraps `load_local_dataset` to build `image_list` and `label_list`.

## Running the notebook

1. Install requirements (TensorFlow 2.x, Keras, NumPy, scikit-learn, matplotlib, tqdm, opencv, mtcnn if using face modules).
2. Execute sections in order:
	- helper utilities
	- MACNN definition
	- data loader (CIFAR-10 or custom)
	- train/test split and client creation
	- IID federated training + plots
	- Non-IID federated training + comparison plots
3. Metrics and figures are written to the working directory (e.g., `fl_iid_comprehensive_analysis.png`).

## Result
<img width="1053" height="576" alt="image" src="https://github.com/user-attachments/assets/b0ac5f78-b052-4b49-b737-d29d94d8390c" />
<img width="1198" height="900" alt="image" src="https://github.com/user-attachments/assets/62407846-ff0e-4809-b1c3-bd966268d641" />
<img width="1314" height="330" alt="image" src="https://github.com/user-attachments/assets/19600095-4159-4106-8802-ac500669a7a5" />


