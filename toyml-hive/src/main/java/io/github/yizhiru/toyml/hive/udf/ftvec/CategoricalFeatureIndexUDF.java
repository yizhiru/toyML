package io.github.yizhiru.toyml.hive.udf.ftvec;

import org.apache.hadoop.hive.ql.exec.UDF;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 离散特征词典索引，相当于tf.feature_column.categorical_column_with_vocabulary_list 函数
 * 词典索引编号从1开始，不在词典中返回默认索引值
 */
public final class CategoricalFeatureIndexUDF extends UDF {

    private static final ConcurrentHashMap<String, Map<String, Integer>> VOCABULARY_CACHE =
            new ConcurrentHashMap<>();

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
        if (value == null || vocabulary == null || vocabulary.isEmpty()) {
            return defaultIndex;
        }

        Map<String, Integer> indexMap = VOCABULARY_CACHE.computeIfAbsent(
                buildCacheKey(vocabulary),
                key -> buildIndexMap(vocabulary));
        return indexMap.getOrDefault(value, defaultIndex);
    }

    private static String buildCacheKey(List<String> vocabulary) {
        return String.join("\u0000", vocabulary);
    }

    private static Map<String, Integer> buildIndexMap(List<String> vocabulary) {
        Map<String, Integer> indexMap = new HashMap<>(vocabulary.size());
        for (int i = 0; i < vocabulary.size(); i++) {
            indexMap.put(vocabulary.get(i), i + 1);
        }
        return indexMap;
    }
}
