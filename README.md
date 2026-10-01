## CAFNet_CD
Contour-Guided Triple Attention Fusion Network for SAR Image Change Detection

## Introduction
Synthetic Aperture Radar (SAR) image change detection is a critical task for monitoring surface changes in remote sensing applications. However, existing SAR change detection methods primarily rely on spatial and channel attention mechanisms, often overlooking the special attention of change contours. To address this, we introduce a contour-guided triple attention fusion network (CAFNet). This framework integrates channel and spatial attention while using contour information to guide the comparison of structural feature. Specifically, the proposed hybrid attention module (HAM) extracts both global context and local details through the coordinated operation of channel and spatial attention. Furthermore, we design a contour-aware attention module (CAAM) that embeds the parameter-free Sobel operator as a structural microfilter within a learnable attention framework, enabling precise estimation of change contours. Experimental evaluation on three real SAR datasets confirms the effectiveness of CAFNet, which achieves PCC values of 96.73%, 98.41%, and 96.84% on the respective datasets, demonstrating its superior detection performance.

## Citation
If you use this code for your research, please cite our paper. Thank you!

@ARTICLE{**, author={Chunyang Li, Jun Wang, Sanku Niu, Xiangyu Yang.}, journal={IEEE Geoscience and Remote Sensing Letters}, title={Contour-Guided Triple Attention Fusion Network for SAR Image Change Detection}, year={2026}.}

## Running
Run the CHG_CD demo files (tested in Matlab 2024b)!

If you have any queries, please contact me (36110@qzc.edu.cn).
