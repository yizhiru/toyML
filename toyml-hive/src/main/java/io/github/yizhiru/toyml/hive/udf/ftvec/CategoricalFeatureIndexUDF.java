package io.github.yizhiru.toyml.hive.udf.ftvec;

import org.apache.hadoop.hive.ql.exec.UDF;

import java.util.List;

/**
 * 离散特征词典索引，相当于tf.feature_column.categorical_column_with_vocabulary_list 函数
 * 词典索引编号从1开始，不在词典中返回默认索引值
 */
public final class CategoricalFeatureIndexUDF extends UDF {

    /**
     * 离散特征词典索引
     *
     * @param value        离线特征值
     * @param vocabulary   离散特征词典
     * @param defaultIndex 默认索引值，若特征值不在词典中，则返回默认索引值
     * @return 词典索引值
     */
    public int evaluate(String value,
                        List<String> vocabulary,
                        int defaultIndex) {
        if (value == null || vocabulary == null) {
            return defaultIndex;
        }

        int index = vocabulary.indexOf(value);
        if (index == -1) {
            return defaultIndex;
        }
        return index + 1;
    }
}
