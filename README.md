### Joint Depth Map Super-Resolution Method via Deep Hybrid-Cross Guidance Filter (Pattern Recognition2023)
#### Authors：Ke Wang, Lijun Zhao, Jinjing Zhang, Jialong Zhang, Anhong Wang, Huihui Bai

### Abstract: 
Nowadays color guided depth Super-Resolution (SR) methods mainly have three thorny problems: (1) joint depth SR methods have serious detail and structure loss at very high sampling rate; (2) existing depth SR networks have high computational complexity; (3) color-depth inconsistency makes it hard to fuse dual-modality features. To resolve these problems, we propose Joint hybrid-cross Guidance Filter (JGF) method to progressively recover the quality of degraded Low-Resolution (LR) depth maps by exploiting color-depth consistency from multiple perspectives. Specifically, the proposed method leverages pyramid structure to extract multi-scale features from High-Resolution (HR) color image. At each scale, hybrid side window filter block is proposed to achieve high-efficiency color feature extraction after each down-sampling for HR color image. This block is also used to extract depth features from LR depth maps. Meanwhile, we propose a multi-perspective cross-guided fusion filter block to progressively fuse high-quality multi-scale structure information of color image with corresponding enhanced depth features. In this filter block, two kinds of space-aware group-compensation modules are introduced to capture various spatial features from different perspectives. Meanwhile, color-depth cross-attention module is proposed to extract color-depth consistency features for impactful boundary preservation. Comprehensively qualitative and quantitative experimental results have demonstrated that our method can achieve superior performances against a lot of state-of-the-art depth SR approaches in terms of mean absolute deviation and root mean square error on Middlebury, NYU-v2 and RGB-D-D datasets.

### Our training codes are publicly available [here](https://github.com/mdcnn/JGF2022/blob/main/JGF-8x.rar).

### Testing and Training Datasets:
You can download datasets by clicking .

### The JGF Result.
Please click [here](https://drive.google.com/file/d/11k2G5QHQuoGcDJcO2DE9Gasi9t1RbTg2/view?usp=sharing) to download results. 

## Supplementary Material 
[Link1](https://github.com/mdcnn/JGF2022), [Link2](https://wa01gy6lnb.feishu.cn/file/boxcn4VYBaQMnPAXwaosJy8xJjn) and [Link3](https://drive.google.com/file/d/1loGL7JBC_dCIgbQSkdlhaX8ESb3UeFPo/view?usp=sharing)

## Citation
If you find our work useful for your research, please cite us:
```
@article{wang2023joint,
  title={Joint depth map super-resolution method via deep hybrid-cross guidance filter},
  author={Wang, Ke and Zhao, Lijun and Zhang, Jinjing and Zhang, Jialong and Wang, Anhong and Bai, Huihui},
  journal={Pattern Recognition},
  volume={136},
  pages={109260},
  year={2023},
  publisher={Elsevier}
}
@inproceedings{zhang2023explainable,
  title={Explainable Unfolding Network For Joint Edge-Preserving Depth Map Super-Resolution},
  author={Zhang, Jialong and Zhao, Lijun and Zhang, Jinjing and Wang, Ke and Wang, Anhong},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={888--893},
  year={2023},
  organization={IEEE}
}
```
