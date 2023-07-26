# A Wolf in Sheep's Clothing: Query-Free Evasion Attacks Against Machine Learning-Based Malware Detectors with Generative Adversarial Networks

This repository contains the code to generate adversarial malware examples that resemble goodware in the feature space. 
The approach is detailed in the paper "[A Wolf in Sheep's Clothing: Query-Free Evasion Attacks Against Machine Learning-Based Malware Detectors with Generative Adversarial Networks](https://arxiv.org/abs/2306.09925)" presented at [WORMA'23](https://worma.gitlab.io/2023/), collocated with [IEEE European Symposium on Security and Privacy 2023](https://eurosp2023.ieee-security.org/).

The code base is not longer maintained and exists as a historical artificat to supplement the paper.

## Dependencies installation

* Python 3.6+
* PyTorch
* LIEF
* EMBER
* Solver dependencies

## Steps to reproduce the code

### Preprocess the executables
To train the GANs you need to preprocess the raw binary programs and extract a numpy feature vector for each type of features. Store the .npz files under /adv_mlw_examples_generation_with_gans/data/npz/ 

### Train the target detection systems
Train the following detectors:
* Byte unigram detector
* API-based detector
* Hashed API-based detector
* String-based detector
* EMBER detector
* MalConv detector

## Acknowledgements
This project has received funding Enterprise Ireland and the European Union’s Horizon 2020 Research and Innovation Programme under Marie Skłodowska-Curie grant agreement No 847402.


