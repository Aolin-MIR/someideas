737trCtEr9tk58tkr1n36v7S7vnM7btU77nfn3tb


# someideas
composing: 
1. 惩罚项 W 小二度音 
2. 特征
3. 不量化 （回归）
4. 采样方式，参考2
5. metric
6. 分开生成各个声部


Text Classification is consisted of sentimental classification (Eprstmt: E-commerce Product Review Dataset for
Sentiment Analysis), news title classification (Tnews: Toutiao Short Text Classification for News), app description
classification (Iflytek: Long Text classification), and subject classification (Csldcp: Chinese scientific literature subjects
classification).
Eprstmt is a binary classification with positive and negative product reviews. Tnews, Iflytek and Csldcp are multi-class
classification with 15, 118 and 67 categories respectively. On tasks with labels as 0 or 1 or in English, we assign each
label with a semantically Chinese meaningful name. For labels longer than one token, we convert those into one-token
labels with the same meaning. For all text classification tasks, label is appended to the end of a sentence, connected
with prompt words. Our generative model predicts the label based on a given sentence, and calculate the probability
P(label|sentence) of each candidate. The candidate with the largest probability is selected as the prediction.

1.和专家讨论总结了，音色迁移模型目前存在的问题。即任务目标难度较大，造成拟合困难。声码器拟合的不够好，且在没见过的乐器上失效。针对以上几点问题，我重新训练了模型。首先降低任务目标难度。
2.比较实验完成，基于扩散模型的decoder效果优于transformer。今后的实验可以以扩散模型作为baseline。但是这个模型结构训练速度较慢。年后我将代码部署在学校的gpu服务器上。
3.小论文计划2开始论证调研，以防目前的成果无法得到发表。这个课题为京剧换装。通过修改目前的衣物目标检测模型(clothes segmentation)和try-on模型分两个步骤实现。阅读了OpenPose[1] , Densepose[3],  CP-VTON [4], CP-VTON+ [5], CP-VTON*[4], PFAFN [6]	, VITON-GT [7], WUTON [8], ACGPN [9].以上工作可以很好地解决普通服饰的替换。输入是衣物的正面图，和人物照片。输出是人物穿上该服装的照片。其中最新的dresscode同时对裤子和配饰也有了更好的支持。

[1] Cao, et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." IEEE TPAMI, 2019.

[2] Li, et al. "Self-Correction for Human Parsing." arXiv, 2019.

[3] Güler, et al. "Densepose: Dense human pose estimation in the wild." CVPR, 2018.

[4] Wang, et al. "Toward Characteristic-Preserving Image-based Virtual Try-On Network." ECCV, 2018.

[5] Minar, et al. "CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On." CVPR Workshops, 2020.

[6] Ge, et al. "Parser-Free Virtual Try-On via Distilling Appearance Flows." CVPR, 2021.

[7] Fincato, et al. "VITON-GT: An Image-based Virtual Try-On Model with Geometric Transformations." ICPR, 2020.

[8] Issenhuth, el al. "Do Not Mask What You Do Not Need to Mask: a Parser-Free Virtual Try-On." ECCV, 2020.

[9] Yang, et al. "Towards Photo-Realistic Virtual Try-On by Adaptively Generating-Preserving Image Content." CVPR, 2020.

