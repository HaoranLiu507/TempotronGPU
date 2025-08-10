## Tempotron on GPU (PyTorch)

### Overview

This repository contains a GPU-accelerated implementation of the Tempotron, a supervised spiking neuron model for spike-timing-based classification [1,2]. The implementation leverages PyTorch to speed up spike encoding, training, and inference, achieving over 500× speedups compared to a CPU baseline in our experiments.

- **Domain demo**: binary pulse-shape discrimination (neutron vs. gamma) [3,4]
- **Generalizable**: the encoder and learning rule can be applied to other domains (e.g., audio, sensor signals)
- **Features**: latency + Gaussian receptive field encoding, noise augmentation (Gaussian/jitter/adding&missing), momentum, mini-batches, variable learning rate

Read the full technical description in our paper: [IEEE TNS (2024)](https://doi.org/10.1109/TNS.2024.3444888) and the preprint on [arXiv](https://doi.org/10.48550/arXiv.2305.18205).

If this work helps your research, please cite the paper (see Citation below).

### Dataset

We use 9,404 radiation pulse signals (≈2k neutron, >7k gamma), length ~280 ns, provided as comma-separated `.txt` files.

Download: [Zenodo DOI: 10.5281/zenodo.7974151](https://doi.org/10.5281/zenodo.7974151)

Expected directory layout (place under the repository root):

```
Dataset/
  dataset/
    training_dataset/
      training_data_normalized.txt
      train_labels.txt
    validation_dataset/
      validation_data_normalized.txt
      test_labels.txt
```

Quick check (optional) to load validation data with NumPy:

```python
import numpy as np, os
val_x = np.loadtxt(os.path.join('Dataset','dataset','validation_dataset','validation_data_normalized.txt'),
                   dtype=np.float32, delimiter=',')
val_y = np.loadtxt(os.path.join('Dataset','dataset','validation_dataset','test_labels.txt'),
                   dtype=np.float32, delimiter=',')
print(val_x.shape, val_y.shape)
```

### Installation

We recommend Python 3.8 with CUDA 11.8 and PyTorch 2.0.0 (newer versions generally work).

```
# create and activate an environment (example with conda)
conda create -n tempotron python=3.8 -y
conda activate tempotron

# install matplotlib
pip install matplotlib

# install PyTorch matching your CUDA toolkit
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Notes
- Apple Silicon MPS in PyTorch 2.0.0 is unstable; use with caution.
- The code defaults to GPU at `cuda:0`. If you need CPU, change the `device` line in `Tempotron.py`:

```python
device = th.device("cuda:0")  # change to: th.device("cpu") or auto: th.device("cuda" if th.cuda.is_available() else "cpu")
```

### Repository layout

- `Tempotron.py`: Tempotron model and data encoding utilities (train/test/validate)
- `train_main.py`: CLI entry to train the model
- `validation_main.py`: Evaluate a saved set of efficacies on the validation set
- `Efficacies/`: saved synaptic efficacies per epoch (`Epoch{N}_efficacies.pt`)
- `Loss/`: training/validation loss tensors and `loss_curves.png`

### Quickstart

1) Prepare the dataset under `Dataset/` as shown above.

2) Train:

```
python train_main.py \
  --dendrites_num 25 \
  --epoch 300 \
  --batchsize 5 \
  --noise_key "Gaussian, jitter, adding&missing" \
  --Gaussian_sigma 1e-4 \
  --jitter_sigma 1e-4 \
  --adding_missing_prob 1e-4
```

Outputs
- Efficacies per epoch in `Efficacies/` (e.g., `Epoch0_efficacies.pt`, …, `Epoch299_efficacies.pt`)
- Loss curves and tensors in `Loss/` (`train_loss.pt`, `test_loss.pt`, `loss_curves.png`)

Note: During training, `test_loss.pt` reflects performance on an internal 20% hold-out split from `training_dataset`. The separate `validation_dataset` is used only by `validation_main.py`.

3) Validate:

```
python validation_main.py
```

By default `validation_main.py` loads `Efficacies/Epoch299_efficacies.pt`. To validate a different epoch, edit `EFFICACIES_NAME` in `validation_main.py` (and ensure `dendrites_num` matches the saved efficacies).

### Command-line options (training)

- `--dendrites_num` (int, default 25): number of dendrites (afferents)
- `--epoch` (int, default 300): training epochs
- `--batchsize` (int, default 5): batch size
- `--noise_key` (str): any combination of `Gaussian, jitter, adding&missing` or `off`
- `--Gaussian_sigma` (float, default 1e-4)
- `--jitter_sigma` (float, default 1e-4)
- `--adding_missing_prob` (float, default 1e-4)

Learning rate schedule is defined in `train_main.py` as an interval `[1e-6, 1e-3]` whose upper bound halves the gap to the lower bound every epoch/20 (~5% of total epochs; for 300 epochs, every 15 epochs).

Tips
- For `threshold=1`, use `dtype=th.float64` for efficacies (already set). Lower precision may require a higher threshold.
- To reproduce exactly, set a manual seed in `train_main.py` (the example includes a commented `th.manual_seed(723)`).

### Programmatic use

```python
import torch as th
import Tempotron as Tp

# load saved efficacies (example)
eff = th.load('Efficacies/Epoch299_efficacies.pt')
tempotron = Tp.Tempotron(V_rest=0, tau=8.4, tau_s=2.1, synaptic_efficacies=eff,
                         A=1, dendrites_num=25, echo=1, threshold=1.0)

# evaluate on the validation set in batches of 10
err = tempotron.test_batch(batchsize=10)
print('Accuracy: {:.2f}%'.format((1 - err) * 100))
```

### Citation

Hao-Ran Liu, Peng Li, Ming-Zhe Liu, Kai-Ming Wang, Zhuo Zuo, and Bing-Qi Liu. **Pulse shape discrimination based on the Tempotron: a powerful classifier on GPU.** *IEEE Transactions on Nuclear Science*, 71(10):2297–2308, Oct. 2024. doi: 10.1109/TNS.2024.3444888. [IEEE](https://doi.org/10.1109/TNS.2024.3444888) · [arXiv](https://doi.org/10.48550/arXiv.2305.18205)

### References

[1] Gütig, Robert, and Haim Sompolinsky. The tempotron: a neuron that learns spike timing–based decisions. *Nature Neuroscience* 9(3), 420–428 (2006).

[2] Gütig, Robert, and Haim Sompolinsky. Tempotron learning. In: *Encyclopedia of Computational Neuroscience*. Springer (2014).

[3] Liu, Hao-Ran, et al. Discrimination of neutrons and gamma rays in plastic scintillator based on pulse-coupled neural network. *Nuclear Science and Techniques* 32(8), 82 (2021).

[4] Liu, Hao-Ran, et al. Discrimination of neutron and gamma ray using the ladder gradient method and analysis of filter adaptability. *Nuclear Science and Techniques* 33(12), 159 (2022).

### Acknowledgements

This project draws on techniques from the CPU-based Tempotron implementation at `https://github.com/dieuwkehupkes/Tempotron`.

### License

This project is released under the terms of the license in `LICENSE`.

### Contact

Please open a GitHub issue or email `liuhaoran@cdut.edu.cn`.

