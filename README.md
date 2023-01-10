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


