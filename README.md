# toyML

工业界常见机器学习算法的教学/参考实现，覆盖排序、CTR 预估、重排序、召回，以及 Hive 离线特征 UDF 与 Spark 数据导出任务。

## 项目结构

```
toyML/
├── toyml/              # Python 模型库（TensorFlow / Keras）
├── toyml-spark/        # Spark 离线任务（召回、TFRecord 导出）
├── toyml-hive/         # Hive UDF（特征哈希、词典索引、分箱）
└── examples/           # 各算法示例脚本
```

## 算法模块

### Learning to Rank
- LambdaGBDT（`examples/ltr_xgb.py`）
- LambdaDNN（`examples/ltr_tfr.py`）
- TFRecord 转换：`toyml/ltr/helper.py`
- NDCG 评估：`toyml/ltr/metrics.py`（基于 scikit-learn）

### CTR 预估
- LR（Logistic Regression）：`toyml/ctr/lr/logistic_regression.py`（Keras 实现）
- XGBoost + LR：`toyml/ctr/lr/gbdt_lr.py`
- DIN（Deep Interest Network）：`toyml/ctr/din/din.py`（含 Local Activation Unit）

### Re-ranking 重排序
- LambdaDNN（`examples/reranking_dnn.py`）
- DLCM（`examples/reranking_dlcm.py`）
- PRM（`examples/reranking_prm.py`）

### 召回
- itemCF（Spark）：`toyml-spark/.../ItemCFSim.scala`
- Node2Vec（Spark）：`toyml-spark/.../Node2VecSim.scala`
- YouTubeDNN：`toyml/match/youtube_dnn/`
- SBCNM（Sampling-Bias-Corrected Neural Model）：`toyml/match/sbcnm/`
- DSSM（Deep Structured Semantic Model）：`toyml/match/dssm/`

### Hive UDF（对应 tf.feature_column）
| UDF | 等价 API |
|-----|----------|
| `CategoricalFeatureHashUDF` | `categorical_column_with_hash_bucket` |
| `CategoricalFeatureIndexUDF` | `categorical_column_with_vocabulary_list` |
| `FeatureBinningUDF` | `bucketized_column` |

## 环境依赖

### Python

```bash
pip install -r toyml/requirements.txt
```

主要依赖：TensorFlow 2.x、tensorflow-ranking、scikit-learn、xgboost、tf-models-official（PRM Transformer）。

运行示例前，将项目根目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/toyML:$PYTHONPATH
```

### Spark（toyml-spark）

- Spark 2.4 / Scala 2.11
- 构建：`cd toyml-spark && mvn package`

### Hive（toyml-hive）

- Hive 4.x / Hadoop 3.x
- 构建：`cd toyml-hive && mvn package`

## 测试

```bash
# Hive UDF 单元测试（JUnit，含断言）
cd toyml-hive && mvn test

# Spark 模块（ScalaTest 用例目前未被 Surefire 执行，需后续接入 scalatest-maven-plugin）
cd toyml-spark && mvn test
```

Python 模块暂无自动化单元测试，可参考 `examples/` 目录手动验证。

## 近期优化摘要

### 正确性
- 修复 LTR TFRecord 写入遗漏最后一组 query、未关闭 writer 的问题
- 补全 DSSM / SBCNM / PRM 的二元交叉熵损失
- DIN 改用论文中的 Local Activation Unit（LAU）替代通用 Attention
- SBCNM 增加可学习的 log-Q 采样偏差校正层

### 性能
- 召回/排序模型中用 `tf.tile` 替代低效的 `tf.gather` 广播
- LR 迁移至 Keras；NDCG 改用 scikit-learn，去除 TF1 Session
- Hive 分箱 UDF 改为二分查找；词典索引 UDF 增加 vocab 缓存
- Node2Vec 相似度计算改为 executor 端分布式计算
- TFRecord 导出支持 `--numPartitions`（默认 8），修复 CTR 输出遗漏 `click_cid4_hist`

### 工程化
- 统一 Python 包导入路径（`toyml.*`）
- 锁定 `requirements.txt` 版本，补充缺失依赖
- Spark/Hive `pom.xml` 补充 `commons-math3`、`RoaringBitmap`、Guava

## 兼容性说明

| 变更 | 影响 |
|------|------|
| LR 模型格式 | 由 TF1 SavedModel 改为 Keras 格式，旧模型需重新训练 |
| `GBDTLRClassifier.fit()` | `X_test` 改为可选参数，仅训练集即可 fit |
| `tf_ndcg()` | 实现改为 scikit-learn，数值可能与原 TF Ranking 版本略有差异 |
| SBCNM 示例 | 损失函数由 `cross_entropy_loss` 改为 `sbcnm_loss` |

## 示例

```bash
# CTR - DIN
python examples/ctr_din.py

# 召回 - SBCNM
python examples/match_sbcnm.py --train_path=train.tfrecord --eval_path=test.tfrecord

# LTR - XGBoost
python examples/ltr_xgb.py
```

## 已知限制

- ItemCF 仍使用用户维度 self-join，超长行为序列时 shuffle 开销大
- Node2Vec 相似度为暴力 top-K，超大 item 库需进一步优化（ANN 等）
- Spark 2.4 / Scala 2.11 栈较旧，生产环境建议评估升级至 Spark 3.x
- `toyml-spark` 的 ScalaTest 用例尚未接入 CI 执行

## License

见 [LICENSE](LICENSE) 文件。
