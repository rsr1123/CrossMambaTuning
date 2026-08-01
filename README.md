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

<div align="center">
<img src="./assets/framework.png" width="95%"/>
</div>

The current release provides the CrossMambaTuning network architecture and task configurations. Test code, training code, and pretrained weights will be released in subsequent updates.

## 📊 Experimental Results

<div align="center">
<img src="./assets/RD.png" width="95%"/>
</div>

Performance comparison on image classification, object detection, and instance segmentation using TIC as the base codec. We report BD-rate and BD-acc/mAP; the best and second-best results are highlighted in **bold** and <u>underline</u>, respectively.

| Method | Venue | Classification<br>BD-rate ↓ | Classification<br>BD-acc ↑ | Detection<br>BD-rate ↓ | Detection<br>BD-mAP ↑ | Segmentation<br>BD-rate ↓ | Segmentation<br>BD-mAP ↑ | Trainable Params ↓ (M) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full fine-tuning | — | / | 17.688 | -73.943% | 4.511 | -67.977% | 3.755 | 7.51 (100.00%) |
| Channel Selection | ICPR'22 | -37.178% | 6.278 | 6.849% | -0.550 | 16.511% | -0.949 | 0.92 (12.25%) |
| ICMH-Net | ACM MM'23 | -18.759% | 3.360 | -9.080% | 0.625 | -10.772% | 0.654 | 3.98 (53.00%) |
| TransTIC | ICCV'23 | -58.529% | 9.956 | -46.301% | 2.768 | -46.521% | 2.690 | 1.62 (21.57%) |
| Adapt-ICMH | ECCV'24 | -88.573% | 16.901 | -55.150% | 3.547 | -52.407% | 3.208 | 0.29 (3.86%) |
| SVD-LoRA | CVPR'25 | -50.162% | 7.920 | -39.927% | 2.207 | -42.431% | 1.938 | <u>0.09 (1.20%)</u> |
| Ours-Tiny | — | -83.187% | 16.118 | -58.236% | 3.742 | -55.387% | 3.266 | **0.08 (1.07%)** |
| Ours-Small | — | <u>-91.570%</u> | <u>16.934</u> | <u>-60.575%</u> | <u>3.980</u> | <u>-60.661%</u> | <u>3.426</u> | 0.15 (2.00%) |
| Ours-Base | — | **-92.788%** | **17.575** | **-65.607%** | **4.249** | **-62.589%** | **3.624** | 0.32 (4.26%) |

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
