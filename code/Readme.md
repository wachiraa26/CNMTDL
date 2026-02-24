# MTDL-CNIA
This is repository contains the code implementation of Manifold Topological Deep Learning via Combinatorial Networks for Image Analysis

## Python Environment
Python 3.10.12
medmnist 3.0.2
h5py 3.15.0
numpy 2.2.6
pandas 2.3.3
pillow 10.4.0
scipy  1.15.3
torch   2.7.1+cu118
pytorch  2.6.0
torch-geometric   2.7.0
CUDA  11.8.87

## MTDL-CNIA model architecture


The source code is located in the 'code/' directory.

The implementation consists of two main stages:
1. **Image Decomposition Stage**  
2. **Image Analysis (CCNN Training) Stage**


### Image Decomposition
Core decomposition functions are implemented in: decomposition.py

To decompose images into the three orthogonal components (gradient, curl, and harmonic components), run:

python prepare_decomposition.py

This will generate the decomposed dataset (e.g., BloodMNIST).

### Model Training and Evaluation
To train and evaluate the Combinatorial Convolutional Neural Network (CCNN), run: 

python main.py
