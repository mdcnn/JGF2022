# Joint Depth Map Super-Resolution Method via Deep Hybrid-Cross Guidance Filter (Ke Wang, Lijun Zhao, Jinjing Zhang, Jialong Zhang, Anhong Wang, Huihui Bai)

Nowadays color guided depth Super-Resolution (SR) methods mainly have three thorny problems: (1) joint depth SR methods have serious detail and structure loss at very high sampling rate; (2) existing depth SR networks have high computational complexity; (3) color-depth inconsistency makes it hard to fuse dual-modality features. To resolve these problems, we propose Joint hybrid-cross Guidance Filter (JGF) method to progressively recover the quality of degraded Low-Resolution (LR) depth maps by exploiting color-depth consistency from multiple perspectives. Specifically, the proposed method leverages pyramid structure to extract multi-scale features from High-Resolution (HR) color image. At each scale, hybrid side window filter block is proposed to achieve high-efficiency color feature extraction after each down-sampling for HR color image. This block is also used to extract depth features from LR depth maps. Meanwhile, we propose a multi-perspective cross-guided fusion filter block to progressively fuse high-quality multi-scale structure information of color image with corresponding enhanced depth features. In this filter block, two kinds of space-aware group-compensation modules are introduced to capture various spatial features from different perspectives. Meanwhile, color-depth cross-attention module is proposed to extract color-depth consistency features for impactful boundary preservation. Comprehensively qualitative and quantitative experimental results have demonstrated that our method can achieve superior performances against a lot of state-of-the-art depth SR approaches in terms of mean absolute deviation and root mean square error on Middlebury, NYU-v2 and RGB-D-D datasets.

# Submitted to Pattern Recognition (Major Revision)

# Testing and Training Datasets:
You can download datasets by clicking [here](https://wa01gy6lnb.feishu.cn/drive/folder/fldcnJamfZfeiAAcvjD26CibsHd).

# Train: 

```
python train.py
```
# Test:

```
python test.py
```

## The JGF Result.
Please click [here](https://drive.google.com/file/d/11k2G5QHQuoGcDJcO2DE9Gasi9t1RbTg2/view?usp=sharing) to download results. 
  
## Available codes
You can download our codes by clicking [here](https://drive.google.com/file/d/1pkWJ1J73gOpdxjEiaZg6SNEj92-wCil3/view?usp=sharing).
