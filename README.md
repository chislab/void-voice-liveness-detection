# void-voice-liveness-detection

Reproduction of paper Void: A Fast and Light Voice Liveness Detection System

## 任务描述
- 受到2020 USENIX Security论文“Void: A fast and light voice liveness detection system”的启发，尝试复现文中描述的轻量化语音活体检测方法；
- 任务 1：从语音样本中提取由4种子特征构成的Void语音特征向量；
- 任务 2：训练合适的SVM分类器，用于准确区分真人语音和仿冒攻击语音；

## 数据集
- 主要基于ASVspoof 2017 Version 2.0数据集；
- 原文中为了验证Void系统的活体检测效果，作者还引入了自建重放攻击数据集以及隐藏语音、超声语音指令、合成语音等多种攻击方式；

## 代码功能简介
- data_preparation.py
    - 从[ASVspoof 2017 V2数据集](https://datashare.is.ed.ac.uk/handle/10283/3055)的训练集、开发集和测试集的语音中分别提取Void特征并保存在features_labels路径下；
    - 特征提取相关函数在feature_extraction.py中定义；
- train_svm.py
    - 特征提取完成后，使用训练集和开发集中的特征向量训练SVM分类器；
    - 根据原文中的实验结果，采用RBF核的SVM取得了最低的等错误率（EER=11.6%）；
- svm_evaluation.py
    - 用于单独验证一个已训练好的SVM分类器在测试集数据上的效果；

## 效果
- 目前该项目在ASVspoof 2017 V2测试集上的等错误率与原文中的指标仍存在较大差距；
- 该项目会持续改进，同时也欢迎大家针对特征提取过程和模型训练过程提出宝贵建议；

## 参考文献
- [Ahmed M E, Kwak I Y, Huh J H, et al. Void: A fast and light voice liveness detection system[J]. 2020.](https://www.usenix.org/system/files/sec20-ahmed-muhammad_0.pdf)
- Kinnunen, Tomi; Sahidullah, Md; Delgado, Héctor; Todisco, Massimiliano; Evans, Nicholas; Yamagishi, Junichi; Lee, Kong Aik. (2018). The 2nd Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2017) Database, Version 2, [sound]. University of Edinburgh. The Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2332.
