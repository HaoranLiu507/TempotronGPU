    # Tempotron on GPU

## Introduction

This project showcases an implementation of the Tempotron, a supervised synaptic learning algorithm developed by Robert Gütig and Haim Sompolinsky [1]. This implementation leverages the robust computational capabilities of the Graphics Processing Unit (GPU), significantly accelerating the spike pulse encoding, training, and classification. By utilizing the Tempotron learning method based on a gradient-driven rule, an integrate-and-fire neuron can effectively categorize a wide array of input classes, provided they are encoded with precise spike timing [2]. While originally conceived for neutron and gamma-ray pulse shape discrimination—a binary classification task in the field of nuclear science [3,4]—this project can be adapted to address various classification issues by making adjustments to the input signal, such as for image and voice classification. Additionally, this project draws on many techniques from the Tempotron project "https://github.com/dieuwkehupkes/Tempotron", which applied a basic Tempotron model on CPU. For a complete description of the current project, including input signal encoding schemes, Tempotron learning rule, and GPU computation implementation, please read our research paper at “DOI: 10.1109/TNS.2024.3444888)”.

If you find our work useful in your research or publication, please cite our work:

Hao-Ran Liu, Peng Li, Ming-Zhe Liu, Kai-Ming Wang, Zhuo Zuo, and Bing-Qi Liu, **"Pulse shape discrimination based on the Tempotron: a powerful classifier on GPU."** *in IEEE Transactions on Nuclear Science,* vol. 71, no. 10, pp. 2297-2308, Oct. 2024, doi: 10.1109/TNS.2024.3444888. </i> [[IEEE](https://doi.org/10.1109/TNS.2024.3444888)] [[arXiv](https://doi.org/10.48550/arXiv.2305.18205)]

## Dataset

This project uses a dataset consisting of 9,404 radiation pulse signals of neutrons and gamma-rays, including approximately 2,000 neutron events and over 7,000 gamma ray events. Each pulse signal has a length of about 280 ns, and the dataset is saved in txt format, which can be read using the loadtxt function supported by the NumPy library.

Please download the dataset from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7974151.svg)](https://doi.org/10.5281/zenodo.7974151)
```
import numpy as np
import os

data = np.loadtxt(
    os.path.abspath(".") + os.sep +
    "Dataset" + os.sep + "dataset" + os.sep + "validation_dataset" + os.sep + "validation_data_normalized.txt",
    dtype=np.float32,
    delimiter=',')

data_labels = np.loadtxt(
    os.path.abspath(
        ".") + os.sep + "Dataset" + os.sep + "dataset" + os.sep + "validation_dataset" + os.sep + "test_labels.txt",
    dtype=np.float32,
    delimiter=','
)
```

## Requirements

To run this project, you will need to have the following libraries installed: matplotlib and PyTorch.


```
# To install matplotlib, run the following code:
pip install matplotlib

# To ensure that Torch is installed correctly, make sure CUDA is avaliable on your device
# Visit CUDA's website, choose the right version for your device (https://developer.nvidia.com/cuda-downloads)

# Then, install the corresponding version of PyTorch
# Visit the PyTorch's website and getting install command (https://pytorch.org/)
# For example, 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
The development environment for this project is python 3.8, cuda 11.8, and torch 2.0.0, and newer versions are expected to work as well.

## Files

This repository contains three Python files:

- `Tempotron.py`: This file implements a class named Tempotron, which includes all necessary attributes and methods to train, test, validate, and fine-tune the Temptron model.
- `train_main.py`: This file is responsible for training the Tempotron model.
- `validation_main.py`: This file performs classification testing on the trained Tempotron model.

## Usage

To use this code, follow these steps:

1. Download or clone this repository.

2. Open the terminal and navigate to the project's root directory.

3. Set model parameters and run the following command to initiate training of the Tempotron model:

   ```
   python train_main.py
   ```

4. The program will output the synaptic efficacies of each epoch in the Directory *Efficacies*, and the losses on the train and testing sets in the Directory *Loss*. 

5. Once training is complete, run the following command to test the Tempotron model:

   ```
   python validation_main.py
   ```

6. The program will output the classification accuracy of the trained Tempotron model on the validation set.

## Conclusion

* One of the most representative third-generation neural network models, the Tempotron, is implemented in this project, with optional random noise augmentation, momentum training, mini-batch training and testing, and variable learning rate.
* Three types of noise are used in random noise augmentation: Gaussian, jitter, and adding&missing, to improve the generalization performance of the Tempotron.
* The computational efficiency of GPU is fully utilized by implementing Tempotron on PyTorch. Compared with the CPU-based Tempotron, this project achieves over 500 times faster in training and testing.

Have fun with the Tempotron on GPU ! :)

If you encounter any problems, please feel free to contact us via *Github Issues*, or simply via email: *liuhaoran@cdut.edu.cn*

## References

[1] Gütig, Robert, and Haim Sompolinsky. **"The tempotron: a neuron that learns spike timing–based decisions."** *Nature neuroscience* 9.3 (2006): 420-428.

[2] Gütig, Robert, and Haim Sompolinsky. **"Tempotron learning."** *Encyclopedia of computational neuroscience*. Springer Science+ Business Media, 2014.

[3] Liu, Hao-Ran, et al. **"Discrimination of neutrons and gamma rays in plastic scintillator based on pulse-coupled neural network."** *Nuclear Science and Techniques* 32.8 (2021): 82.

[4] Liu, Hao-Ran, et al. **"Discrimination of neutron and gamma ray using the ladder gradient method and analysis of filter adaptability."** *Nuclear Science and Techniques* 33.12 (2022): 159.
    
