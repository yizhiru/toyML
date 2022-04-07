package io.github.yizhiru.toyml.hive.udf.ftvec;

import com.google.common.hash.Hashing;
import org.apache.hadoop.hive.ql.exec.UDF;

import java.nio.charset.StandardCharsets;

/**
 * 离散特征Hash，相当于tf.feature_column.categorical_column_with_hash_bucket 函数
 * 特征哈希值属于区间 [1, bucketSize-1]，其中，0值特定为padding用
 */
public final class CategoricalFeatureHashUDF extends UDF {

    /**
     * 特征hash公式：Hash(input_feature_string) % (bucket_size-1) + 1
     *
     * @param value      特征值
     * @param bucketSize 分桶数
     * @return 特征哈希值
     */
    public int evaluate(String value, int bucketSize) {
        if (value == null) {
            value = "";
        }

        int bit = Hashing.murmur3_32()
                .hashString(value, StandardCharsets.UTF_8)
                .hashCode();
        return Math.abs(bit) % (bucketSize - 1) + 1;
    }
}
