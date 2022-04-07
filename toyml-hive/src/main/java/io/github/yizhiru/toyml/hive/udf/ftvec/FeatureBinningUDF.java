package io.github.yizhiru.toyml.hive.udf.ftvec;

import org.apache.hadoop.hive.ql.exec.UDF;

import java.util.List;

/**
 * 连续特征分箱，相当于tf.feature_column.bucketized_column 函数
 */
public class FeatureBinningUDF extends UDF {

    /**
     * 离线特征分箱
     *
     * @param val         离线特征值
     * @param percentiles 分箱的分位点（为右包含关系），小于等于该分位点，则属于该分箱
     * @return [0, len] 分箱值，若值为null，则返回0；若percentiles为null或长度为0，则返回0
     */
    public int evaluate(Double val, List<Double> percentiles) {
        int len = percentiles.size();
        if (len == 0 || val == null) {
            return 0;
        }

        for (int i = 0; i < len; i++) {
            if (val <= percentiles.get(i)) {
                return i;
            }
        }
        return len;
    }
}
