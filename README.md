
![Simple Few Shot](assets/logo.png)

[![PyPI](https://img.shields.io/pypi/v/simplefsl?label=simplefsl&logo=python&style=for-the-badge)](https://pypi.org/project/simplefsl)
[![Colab](https://img.shields.io/badge/Open%20In-Colab-FFD43B?logo=googlecolab&logoColor=black&style=for-the-badge)](https://colab.research.google.com/drive/11m4Dbvgpnm4HVaGsuaXE-mjwuaPY30Ik?usp=sharing)


## Overview

A modular library for **few-shot learning**, implementing classic and state-of-the-art meta-learning algorithms

It's designed to be flexible, extensible, and easy to integrate into your research or production pipeline

- Plug-and-play support for multiple few-shot learning models
- Episodic training and evaluation setup
- Clean dataset interface for custom datasets
- Support for custom backbones like **ResNet**, **ViT**, and more
- Built-in data augmentation options (`basic`, `mixup`, `cutmix`)

## Available Models

The following model results are based on the original papers, evaluated on Mini-Imagenet, using 5-way 5-shot tasks.

| Model           | Paper Reference                                                                  | Input parameter name | [Mini-imagenet accuracy (%)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3) |
|-----------------|----------------------------------------------------------------------------------|----------------------|---------------------------------------------------------------|
| **ProtoNet**    | [https://arxiv.org/abs/1703.05175](https://arxiv.org/abs/1703.05175)             | `proto_net`            | 68.20                                                       |
| **RelationNet** | [https://arxiv.org/abs/1711.06025](https://arxiv.org/abs/1711.06025)             | `relation_net`         | 65.32                                                       |
| **MatchingNet** | [https://arxiv.org/abs/1606.04080](https://arxiv.org/abs/1606.04080)             | `matching_net`         | 60.00                                                       |
| **MetaOptNet**  | [https://arxiv.org/abs/1904.03758](https://arxiv.org/abs/1904.03758)             | `metaopt_net`          | 78.63                                                       |
| **TapNet**      | [https://arxiv.org/abs/1905.06549](https://arxiv.org/abs/1905.06549)             | `tapnet`               | 76.36                                                       |
| **TADAM**       | [https://arxiv.org/abs/1805.10123](https://arxiv.org/abs/1805.10123)             | `tadam`                | 76.70                                                       |
| **DN4**         | [https://arxiv.org/abs/1903.12290](https://arxiv.org/abs/1903.12290)             | `dn4`                  | 71.02                                                       |
| **MSENet**      | [https://arxiv.org/abs/2409.07989v2](https://arxiv.org/abs/2409.07989v2)         | `msenet`               | 84.42                                                       |
| **FEAT**        | [https://arxiv.org/abs/1812.03664](https://arxiv.org/abs/1812.03664)             | `feat`                 | 82.05                                                       |
| **DSN**         | [https://ieeexplore.ieee.org/document/9157772](https://ieeexplore.ieee.org/document/9157772)| `dsn`       | 78.83                                                       |
| **METAQDA**     | [https://arxiv.org/abs/2101.02833](https://arxiv.org/abs/2101.02833)             | `metaqda`              | 84.28                                                       |
| **Negative Margin** | [https://arxiv.org/abs/2003.12060](https://arxiv.org/abs/2003.12060)         | `negativemargin`       | 81.57                                                       |
| **R2D2**        | [https://arxiv.org/abs/1805.08136](https://arxiv.org/abs/1805.08136)             | `r2d2`                 | 68.40                                                       |
| **MAML**        | [https://arxiv.org/abs/1703.03400](https://arxiv.org/abs/1703.03400)             | `maml`                 | 63.10                                                       |
| **MTL**        | [https://arxiv.org/abs/1812.02391](https://arxiv.org/abs/1812.02391)             | `mtl`                   | 74.30                                                       |

All models are implemented as separate `.py` files under the [models/](/simplefsl/models) directory

It is possible to use **ResNet** and **Swin** as backbones for the models. The respective input parameters are `resnet` and `swin`.

## Getting Started

Install the package directly from GitHub:

```bash
pip install git+https://github.com/victor-nasc/SimpleFewShot.git
```

A ready-to-run Colab example is available:

[![Colab](https://img.shields.io/badge/Open%20In-Colab-FFD43B?logo=googlecolab&logoColor=black&style=for-the-badge)](https://colab.research.google.com/drive/11m4Dbvgpnm4HVaGsuaXE-mjwuaPY30Ik?usp=sharing)

## Custom Dataset Format

To use a custom dataset, provide a `.csv` file with the following structure:

- `image_id`: Path to the image file
- One column per class (binary: 1 if image belongs to class, 0 otherwise)

**Example:**

| idx | image_id         | class1 | class2 | class3 |
|-----|------------------|--------|--------|--------|
| 1   | `/data/img1.jpg` | 1      | 0      | 0      | 
| 2   | `/data/img2.jpg` | 0      | 1      | 0      | 
| 3   | `/data/img3.jpg` | 0      | 0      | 1      |
| ... | ...              | ...    | ...    | ...    | 

You can define your training and test class splits programmatically using the `FewShotManager` class

Obs.: This method supports also **multi-labeled** datasets

## Training a model

In order to train a model with your data, run:

```bash
python3 train.py \
  --model <model_name> \
  --backbone <backbone_name> \
  --ways <number_of_classes_per_task> \
  --shots <number_of_examples_per_class_per_task> \
  --gpu <which_gpu_to_run> \
  --lr <lr_value> \
  --l2_weight <l2_weight_value> \
  --augment <cutmix/mixup/None>
```

Model name is required and should be used as informed on the models table, on the `Input parameter name` column. Standard values for the other parameters are given below:

| ways | shots | gpu | lr     | l2_weight |
|------|-------|-----|--------|-----------|
| 2    | 5     | 0   | 0.0001 | 0.0       |

## Generate GradCAM

All heatmaps are generated for images from a class that was not presented during training. Not all models can generate GradCAM explainability, due to their architecture.
In order to generate a GradCAM for an image, run:

```bash
python explain.py \
  --model <model_name> \
  --weights <weights.pth> \
  --ways <number_of_classes_per_task> \
  --shots <number_of_examples_per_class_per_task> \
  --gpu <gpu_index> \
  --image <image_file_name_with_no_extension> \
  --target-class <target_class_name> \
  --classes <"target_class,other_test_class_for_fine_tuning"> \
  --augment <cutmix/mixup/None>
```

## Citation

If you find this library useful in your research or project, please consider citing:
```
@INPROCEEDINGS{11223370,
  author={Ribeiro, Victor N. and Lima, Fernando G. and Michelassi, Rodrigo de C. and Rabelo, Carlos A. M. and Hirata, Nina S. T.},
  booktitle={2025 38th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)}, 
  title={A Comparative Study on Few-Shot Learning for Retinal Disease Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Metalearning;Training;Analytical models;Computational modeling;Retina;Transformers;Libraries;Reproducibility of results;Few shot learning;Diseases},
  doi={10.1109/SIBGRAPI67909.2025.11223370}}
```
