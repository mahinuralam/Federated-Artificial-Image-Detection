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

![image](https://github.com/user-attachments/assets/13f6e507-cc62-41e6-8494-e099b86d5109)

![image](https://github.com/user-attachments/assets/2b5556f4-a42a-49aa-9128-96d292d1f3d3)

![image](https://github.com/user-attachments/assets/4cb3f085-0ed8-4126-8cc0-5a8ab95ee4cd)