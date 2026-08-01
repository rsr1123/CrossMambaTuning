<div align="center">

# 🌟 CrossMambaTuning: Synergistic Spatial and Cross-Layer Adaptation for Machine Vision Compression (ACM MM'26)

**_Synergistic Spatial and Cross-Layer Adaptation for Machine Vision Compression!_**

**Haobo Xiong**, Shaobo Liu, Kai Liu, Chongyang Ding*

*Corresponding author


</div>

## 📌 Abstract

To reduce deployment cost and retraining overhead, adapting pre-trained learned image compression (LIC) models to downstream machine vision tasks has attracted growing attention. However, existing methods typically insert fine-tuning modules independently into frozen backbones, lacking explicit mechanisms for cross-layer coordination. To address this limitation, we propose a novel framework named **CrossMambaTuning**, which integrates State Space Models with cross-layer interaction mechanisms for parameter-efficient fine-tuning. Specifically, we design an efficient Mamba adapter equipped with task-specific prompts and multi-scale branches to precisely capture both local features and global dependencies. Furthermore, we introduce a Scale-Invariant Cross-Layer Adapter (SICA) with a parameter-sharing strategy to fuse task information across different scales and reduce redundancy. Extensive experiments demonstrate that CrossMambaTuning achieves strong performance on multiple machine vision tasks with a small trainable parameter budget.

## 🎯 Highlights

✅ **Mamba Adapter**: Task-specific prompts and multi-scale branches for local and global spatial modeling.<br>
✅ **Cross-Layer Adaptation**: Scale-Invariant Cross-Layer Adapter (SICA) fuses task information across scales with parameter sharing.<br>
✅ **Parameter-Efficient Tuning**: Lightweight adaptation of pre-trained learned image compression models.<br>

## 🕒 Updates

[TODO] Release Tiny/Small network and test code.<br>
[TODO] Release training code and pretrained weights.<br>
[2026/08/01] Initial release of the network architecture and experiment configurations.<br>

## 🚀 Overview

The current release provides the CrossMambaTuning network architecture and task configurations. Test code, training code, and pretrained weights will be released in subsequent updates.

## 📚 Libraries & Dataset

All library versions are listed in `requirements.txt`. For the `selective_scan` dependency, please follow the installation instructions of [VMamba](https://github.com/MzeroMiko/VMamba).

**Datasets**

The following datasets are used in the paper:

- COCO2017 Train/Val for detection and instance segmentation
- Kodak for reconstruction evaluation
- ImageNet for classification

## 📥 Inference: onlyTest

Test / inference code and model checkpoints will be released in a future update.

## 🧩 Example: Train/Eval/Test

Training code and complete reproduction instructions will be released in a future update.

## ⚡ Acknowledgment

Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [VMamba](https://github.com/MzeroMiko/VMamba).
