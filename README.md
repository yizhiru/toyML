# toyML

## 介绍

工业界常见机器学习算法，实现：
- Learning to Rank
  - LambdaGBDT
  - LambdaDNN

- CTR预估
  - LR (Logistic Regression)
  - XGBoost + LR
  - DIN (Deep Interest Network)

- Re-ranking 重排序
  - LambdaDNN
  - DLCM (Deep Listwise Context Model)
  - PRM (Personalized Re-ranking Model)

- 召回
  - itemCF
  - node2vec
  - YouTubeDNN
  - SBCNM (Sampling-Bias-Corrected Neural Model)
  - DSSM (Deep Structured Semantic Model)

Hive-UDF实现tf.feature_column特征处理，包括：
- 离线特征哈希，categorical_column_with_hash_bucket
- 离散特征词典索引，categorical_column_with_vocabulary_list
- 连续特征分箱，bucketized_column





