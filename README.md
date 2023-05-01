 &nbsp;
 &nbsp;
  
<div align="center">
  <img src="assets/negative_contrast.png" width="600"/>
</div>

<div align="center">
	
[![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2109.05565)
[![project-page](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue)](https://opensphere.world/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Negative Contrast** is a hyperspherical face recognition library based on PyTorch. Check out the [project homepage](https://opensphere.world/).
	
</div>

## Introduction
**OpenSphere** provides a consistent and unified training and evaluation framework for hyperspherical face recognition research. The framework decouples the loss function from the other varying components such as network architecture, optimizer, and data augmentation. It can fairly compare different loss functions in hyperspherical face recognition on popular benchmarks, serving as a transparent platform to reproduce published results.


<!-- TABLE OF CONTENTS -->
***Table of Contents***:  - <a href="#setup">Setup</a> - <a href="#get-started">Get started</a> - <a href="#log-and-pretrained-models">Pretrained models</a> - <a href="#reproduce-published-results">Reproducible results</a> - <a href="#citation">Citation</a> -

## Update
- **2023.3.29**: Initial commit.

## Setup
1. Clone the OpenSphere repository. We'll call the directory that you cloned OpenSphere as `$OPENSPHERE_ROOT`.

    ```console
    git clone https://github.com/hiter0/negative contrast.git
    ```

2. Construct virtual environment in [Anaconda](https://www.anaconda.com/):

    ```console
    conda env create -f environment.yml
    ```

## Get started
In this part, we assume you are in the directory `$OPENSPHERE_ROOT`. After successfully completing the [Setup](#setup), you are ready to run all the following experiments.

1. **Download and process the datasets**

  - Download the training set (`VGGFace2`), validation set (`LFW`, `Age-DB`, `CA-LFW`, `CP-LFW`), and test set (`IJB-B` and `IJB-C`) and place them in `data/train`, `data/val` amd `data/test`, respectively.
	
  - For convenience, we provide a script to automatically download the data. Simply run

	```console
	bash scripts/dataset_setup.sh
	```

  - If you need the `MS1M` training set, please run the additional commend:

	```console
	bash scripts/dataset_setup_ms1m.sh
	```
  - To download other datasets (e.g., `WebFace` or `Glint360K`), see the `scripts` folder and find what you need.

2. **Train a model (see the training config file for the detailed setup)**

    We give a few examples for training on different datasets with different backbone architectures:

  - To train SphereFace2 with SFNet-20 on `VGGFace2`, run the following commend (with 2 GPUs):

	```console
	CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train/vggface2_sfnet20_sphereface2.yml
	```

  - To train SphereFace with SFNet-20 on `VGGFace2`, run the following commend (with 2 GPUs):

	```console
	CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train/vggface2_sfnet20_sphereface.yml
	```

  - We provide many config files for training, see [this folder](https://github.com/ydwen/opensphere/tree/main/config/train) for details.

  - After finishing training a model, you will see a `project` folder under `$OPENSPHERE_ROOT`. The trained model is saved in the folder named by the job starting time, eg, `20220422_031705` for 03:17:05 on 2022-04-22.

  - Our framework also re-implements some other popular hyperspherical face recognition methods such as ArcFace, AM-Softmax (CosFace) and CocoLoss (NormFace). Please check out the folder `model/head` and some examplar config files in the folder `config/papers/SphereFace2/sec31`.

3. **Test a model (see the testing config file for detailed setup)**

  - To test on the `combined validation` dataset, simply run

	```console
	CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/test/combined.yml --proj_dir project/##YourFolder##
	```

For more information about how to use training and testing config files, please see [here](https://github.com/ydwen/opensphere/tree/main/config).

## Results and pretrained models

<div align="center">
	
|         Loss          |    Architecture     | Dataset |                             Config & Training Log & Pretrained Model                               |
|:---------------------:|:-------------------:|:---:|:--------------------------------------------------------------------------------------------------:|
 |      SphereFace       |  SFNet-20 (w/o BN)  | VGGFace2 | [Google Drive](https://drive.google.com/file/d/1347KRJDqJiySdAOarVDkSlftg3K-W26g/view?usp=sharing) |
|      SphereFace+      |  SFNet-20 (w/o BN)  | VGGFace2 | [Google Drive](https://drive.google.com/file/d/1CBfxxTN712QmuwTk7i2az6JTs3ZRo-ia/view?usp=sharing) |
 | SphereFace-R (HFN,v2) |  SFNet-20 (w/o BN)  | VGGFace2 | [Google Drive](https://drive.google.com/file/d/1dnylODdnatcVSHitdht8_fN2im-0gEGV/view?usp=sharing) |
 | SphereFace-R (SFN,v2) |  SFNet-20 (w/o BN)  | VGGFace2 |                                            To be added                                             |
</div>
	
## Reproduce published results

We create an additional folder `config/papers` that is used to provide detailed config files and reproduce results in published papers. Currently we provide config files for the following papers:
  
  - SphereFace2: Binary Classification is All You Need for Deep Face Recognition, ICLR 2022


## Citation

If you find **OpenSphere** useful in your research, please consider to cite:

For **SphereFace**:

  ```bibtex
  @article{Liu2022SphereFaceR,
	title={SphereFace Revived: Unifying Hyperspherical Face Recognition},
	author={Liu, Weiyang and Wen, Yandong and Raj, Bhiksha and Singh, Rita and Weller, Adrian},
	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	year={2022}
  }	
  ```

## Contact

  [Yandong Wen](https://ydwen.github.io) and [Weiyang Liu](https://wyliu.com)

  Questions can also be left as issues in the repository. We will be happy to answer them.
